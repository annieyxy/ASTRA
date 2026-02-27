import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_PATH = "libero_90_images/KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it_state0000_clean_no_ketchup_1.png"
PROMPT = (
    "USER: <image>\n"
    "Task: put the butter at the front in the top drawer of the cabinet and close it. "
    "Is the target object present in the scene? Answer YES or NO.\n"
    # "Is ketchup present in the scene? Answer YES or NO.\n"
    "ASSISTANT:"
)



# IMAGE_PATH = "libero_90_images/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_state0000_clutter_no_orange_juice_1.png"
# PROMPT = (
#     "USER: <image>\n"
#     "Task: close the top drawer of the cabinet. "
#     "Is the target object present in the scene? Answer YES or NO.\n"
#     # "Is orange juice present in the scene? Answer YES or NO.\n"
#     "ASSISTANT:"
# )



def _find_image_token_index(input_ids: torch.Tensor, image_token_id: int) -> int:
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    if len(positions) == 0:
        raise ValueError("Could not find <image> token in input_ids.")
    return int(positions[0].item())


def _infer_image_token_span(attn_seq_len: int, text_seq_len: int) -> int:
    # expanded_len = (text_len - 1) + num_image_tokens
    return attn_seq_len - (text_seq_len - 1)


def _find_subsequence(haystack: List[int], needle: List[int]) -> List[int]:
    if len(needle) == 0 or len(needle) > len(haystack):
        return []
    out = []
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            out.append(i)
    return out


def _find_word_occurrences(text_ids: List[int], tokenizer: Any, word: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for variant in [f" {word}", word]:
        needle = tokenizer.encode(variant, add_special_tokens=False)
        for st in _find_subsequence(text_ids, needle):
            spans.append((st, st + len(needle)))
        if spans:
            break
    return spans


def _get_llm_layers(model: LlavaForConditionalGeneration):
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not locate language model decoder layers.")


def _resolve_layer_idx(model: LlavaForConditionalGeneration, layer: int) -> int:
    layers = _get_llm_layers(model)
    return layer if layer >= 0 else (len(layers) + layer)


def _capture_values_from_layer(model: LlavaForConditionalGeneration, layer_idx: int):
    layers = _get_llm_layers(model)
    layer = layers[layer_idx]
    cache: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, out):
        cache["v_proj_out"] = out.detach()

    handle = layer.self_attn.v_proj.register_forward_hook(hook_fn)
    return handle, cache


def _values_to_head_values(v_proj_out: torch.Tensor, model: LlavaForConditionalGeneration) -> torch.Tensor:
    layers = _get_llm_layers(model)
    num_heads = layers[0].self_attn.num_heads
    bs, seq, hidden = v_proj_out.shape
    head_dim = hidden // num_heads
    return v_proj_out.view(bs, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _reduce_heads(attn_hqk: torch.Tensor, head_reduce: str) -> torch.Tensor:
    if head_reduce == "mean":
        return attn_hqk.mean(dim=0)
    if head_reduce == "max":
        return attn_hqk.max(dim=0).values
    raise ValueError("head_reduce must be one of: mean, max")


def _attn_times_value_norm(attn_hqk: torch.Tensor, V_hkd: torch.Tensor, head_reduce: str) -> torch.Tensor:
    vnorm_hk = torch.norm(V_hkd.float(), dim=-1)
    score_hqk = attn_hqk.float() * vnorm_hk[:, None, :]
    return _reduce_heads(score_hqk, head_reduce)


def to_grid(vec: np.ndarray) -> np.ndarray:
    n = int(vec.shape[0])

    def is_square(m: int):
        s = int(math.isqrt(m))
        return s * s == m, s

    ok, s = is_square(n)
    if ok:
        return vec.reshape(s, s)

    for drop_front in [True, False]:
        m = n - 1
        if m > 0:
            ok, s = is_square(m)
            if ok:
                v = vec[1:] if drop_front else vec[:-1]
                return v.reshape(s, s)

    for pad_front in [True, False]:
        m = n + 1
        ok, s = is_square(m)
        if ok:
            pad = np.zeros((1,), dtype=vec.dtype)
            v = np.concatenate([pad, vec]) if pad_front else np.concatenate([vec, pad])
            return v.reshape(s, s)

    for drop_front in [True, False]:
        m = n - 2
        if m > 0:
            ok, s = is_square(m)
            if ok:
                v = vec[2:] if drop_front else vec[:-2]
                return v.reshape(s, s)

    raise ValueError(f"Cannot coerce length {n} into square grid.")


def _resolve_steering_activation_paths() -> Tuple[Path, Path]:
    candidates = [
        Path("activations/llava/distract_steer"),
        Path("distract_steer/activations/llava/distract_steer"),
    ]
    for base in candidates:
        diff_path = base / "diff_activations_train.pt"
        ref_path = base / "reference_activations.pt"
        if diff_path.exists() and ref_path.exists():
            return diff_path, ref_path
    raise FileNotFoundError("Could not find steering activation files.")


def _mean_activation_for_layer(acts_by_wrapper: Dict[str, Dict[int, List[np.ndarray]]], layer: int, device, dtype):
    collected = []
    for _, per_layer in acts_by_wrapper.items():
        if layer in per_layer:
            collected.extend(per_layer[layer])
    if not collected:
        raise ValueError(f"No activations found for layer key {layer}")
    arr = np.array(collected)
    return torch.from_numpy(np.mean(arr, axis=0)).to(device=device, dtype=dtype)


def create_custom_forward_hook(steer_vector: torch.Tensor, reference_vector: torch.Tensor, steer_type: str, alpha: float):
    def custom_forward_hook(_module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        R_feat = hidden.clone()
        target = R_feat[..., -1, :]

        norm_steer_vector = torch.norm(steer_vector, p=2)
        unit_steer_vector = steer_vector / (norm_steer_vector + 1e-12)

        if steer_type == "linear":
            target = target - unit_steer_vector * alpha
        elif steer_type == "projection":
            clip_proj = torch.sum((target - reference_vector) * steer_vector, dim=-1) / (
                torch.norm(target - reference_vector, p=2, dim=-1) * torch.norm(steer_vector, p=2) + 1e-12
            )
            clip_proj = torch.clamp(clip_proj, min=0.0)
            coefficient = clip_proj * torch.norm(target, p=2, dim=-1) * alpha
            target = target - coefficient.unsqueeze(-1) * unit_steer_vector
        elif steer_type != "no_steer":
            raise NotImplementedError(f"Unknown steer_type: {steer_type}")

        R_feat[..., -1, :] = target
        if isinstance(output, tuple):
            return (R_feat,) + output[1:]
        return R_feat

    return custom_forward_hook


def load_model_processor_tokenizer(model_id: str, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = LlavaProcessor.from_pretrained(model_id, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if device == "cuda":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    model.eval()
    return model, processor, tokenizer


def maybe_attach_steering_hook(model: LlavaForConditionalGeneration, steer_layer: int, steer_type: str, steer_alpha: float):
    if steer_type == "no_steer":
        return None

    diff_path, ref_path = _resolve_steering_activation_paths()
    diff_by_wrapper = torch.load(diff_path, map_location="cpu")
    ref_by_wrapper = torch.load(ref_path, map_location="cpu")

    param = next(model.parameters())
    target_device = param.device
    target_dtype = param.dtype

    steer_vector = _mean_activation_for_layer(diff_by_wrapper, steer_layer, target_device, target_dtype)
    reference_vector = _mean_activation_for_layer(ref_by_wrapper, steer_layer, target_device, target_dtype)

    # Notebook uses 1-based activation keys and hooks index (layer-1)
    hook_idx = steer_layer - 1
    layers = _get_llm_layers(model)
    if hook_idx < 0 or hook_idx >= len(layers):
        raise ValueError(f"steer layer index {hook_idx} out of range for {len(layers)} decoder layers")

    return layers[hook_idx].register_forward_hook(
        create_custom_forward_hook(steer_vector, reference_vector, steer_type, steer_alpha)
    )


def _build_prefix_inputs(inputs: Dict[str, torch.Tensor], prefix_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    prefix_inputs = dict(inputs)
    prefix_inputs["input_ids"] = prefix_ids
    prefix_inputs["attention_mask"] = torch.ones_like(prefix_ids, device=prefix_ids.device)
    return prefix_inputs


def _slugify(text: str) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch in (" ", "."):
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "heatmap"


def run_heatmap(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    tokenizer: AutoTokenizer,
    image_path: str,
    prompt: str,
    layer: int,
    head_reduce: str,
    steer_type: str,
    heatmap_type: str,
    target: str,
    keyword: Optional[str],
    which_occurrence: int,
    use_last_subtoken: bool,
    max_new_tokens: int,
    save_dir: str,
):
    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    img_pos_in_text = _find_image_token_index(inputs["input_ids"], image_token_id)

    layer_idx = _resolve_layer_idx(model, layer)

    need_values = heatmap_type == "attn_value"
    handle = None
    cache: Dict[str, torch.Tensor] = {}

    if target == "last_prompt":
        run_inputs = inputs
        text_total_len = inputs["input_ids"].shape[1]
        query_source_desc = "last_prompt"

    elif target == "first_generated":
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                return_dict_in_generate=True,
                output_attentions=False,
                output_scores=False,
                use_cache=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        if gen.sequences.shape[1] <= prompt_len:
            raise RuntimeError("No generated token returned by model.generate")
        prefix_ids = gen.sequences[:, : prompt_len + 1]
        run_inputs = _build_prefix_inputs(inputs, prefix_ids)
        text_total_len = prefix_ids.shape[1]
        generated = tokenizer.decode(gen.sequences[0, prompt_len:], skip_special_tokens=True).strip()
        first_tok = tokenizer.decode([prefix_ids[0, -1].item()])
        print(f"Generated: {generated!r}")
        print(f"First generated token used as query: {first_tok!r}")
        query_source_desc = "first_generated"

    elif target == "keyword":
        if not keyword:
            raise ValueError("keyword target selected but --keyword was not provided")
        run_inputs = inputs
        text_total_len = inputs["input_ids"].shape[1]
        query_source_desc = f"keyword={keyword!r}"

    else:
        raise ValueError(f"Unsupported target: {target}")

    if need_values:
        handle, cache = _capture_values_from_layer(model, layer_idx)

    with torch.no_grad():
        out = model(**run_inputs, output_attentions=True, use_cache=False, return_dict=True)

    if handle is not None:
        handle.remove()

    attn = out.attentions[layer_idx][0]  # (heads, seq, seq)
    _, seq_len, _ = attn.shape

    num_img_tokens = _infer_image_token_span(seq_len, text_total_len)
    img_start = img_pos_in_text
    img_end = img_start + num_img_tokens

    if target == "last_prompt":
        q = seq_len - 1

    elif target == "first_generated":
        q = seq_len - 1

    else:
        text_ids = run_inputs["input_ids"][0].tolist()
        spans = _find_word_occurrences(text_ids, tokenizer, keyword)
        if not spans:
            decoded = tokenizer.decode(text_ids)
            raise ValueError(f"Could not find keyword '{keyword}' in prompt tokens. Decoded prompt:\n{decoded}")
        if which_occurrence >= len(spans):
            raise ValueError(
                f"which_occurrence={which_occurrence} but found only {len(spans)} occurrences for '{keyword}'"
            )

        w_start, w_end = spans[which_occurrence]
        q_text = (w_end - 1) if use_last_subtoken else w_start
        shift = num_img_tokens - 1
        q = q_text + shift if q_text > img_pos_in_text else q_text
        print(f"Matched keyword span (text idx): {(w_start, w_end)}")
        print(f"Query token text idx={q_text}, expanded idx={q}")

    if heatmap_type == "attn":
        score = _reduce_heads(attn, head_reduce)
    else:
        if "v_proj_out" not in cache:
            raise RuntimeError("v_proj output not captured for attn_value mode")
        V = _values_to_head_values(cache["v_proj_out"], model)[0]
        score = _attn_times_value_norm(attn, V, head_reduce)

    img_score = score[q, img_start:img_end].detach().cpu().numpy()
    grid = to_grid(img_score)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid_img = Image.fromarray((grid * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)

    plt.figure(figsize=(7, 7))
    plt.imshow(image)
    plt.imshow(grid_img, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.title(
        f"mode={heatmap_type} | target={query_source_desc} | layer={layer} | q={q} | reduce={head_reduce}"
    )
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = Path(image_path).stem
    prefix = "attn_heatmap"
    file_name = (
        f"{prefix}_steer-{steer_type}_metric-{heatmap_type}_target-{target}_"
        f"layer-{layer}_reduce-{head_reduce}_{image_stem}_{stamp}.png"
    )
    out_path = out_dir / file_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved heatmap to: {out_path}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="LLaVA attention/attn*value heatmap utility")
    p.add_argument("--target", choices=["last_prompt", "first_generated", "keyword"], default="last_prompt")
    p.add_argument("--heatmap-type", choices=["attn", "attn_value"], default="attn_value")
    p.add_argument("--layer", type=int, default=-8)
    p.add_argument("--head-reduce", choices=["mean", "max"], default="mean")

    p.add_argument("--keyword", type=str, default=None)
    p.add_argument("--which-occurrence", type=int, default=0)
    p.add_argument("--use-first-subtoken", action="store_true")

    p.add_argument("--max-new-tokens", type=int, default=10)

    p.add_argument("--model-id", type=str, default=MODEL_ID)
    p.add_argument("--image-path", type=str, default=IMAGE_PATH)
    p.add_argument("--prompt", type=str, default=PROMPT)

    p.add_argument("--steer-layer", type=int, default=16)
    p.add_argument("--steer-type", choices=["no_steer", "linear", "projection"], default="linear")
    p.add_argument("--steer-alpha", type=float, default=40.0)
    p.add_argument("--save-dir", type=str, default="heatmap_outputs_a40")
    # p.add_argument("--save-prefix", type=str, default="attn_heatmap")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor, tokenizer = load_model_processor_tokenizer(args.model_id, device)

    hook = maybe_attach_steering_hook(
        model=model,
        steer_layer=args.steer_layer,
        steer_type=args.steer_type,
        steer_alpha=args.steer_alpha,
    )

    try:
        run_heatmap(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_path=args.image_path,
            prompt=args.prompt,
            layer=args.layer,
            head_reduce=args.head_reduce,
            steer_type=args.steer_type,
            heatmap_type=args.heatmap_type,
            target=args.target,
            keyword=args.keyword,
            which_occurrence=args.which_occurrence,
            use_last_subtoken=not args.use_first_subtoken,
            max_new_tokens=args.max_new_tokens,
            save_dir=args.save_dir,
            # save_prefix=args.save_prefix,
        )
    finally:
        if hook is not None:
            hook.remove()


if __name__ == "__main__":
    main()
