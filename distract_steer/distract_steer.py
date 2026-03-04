import argparse
import contextlib
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, LlavaForConditionalGeneration, LlavaProcessor


DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_IMAGE_DIR = Path("libero_90_images/new_by_category")
DEFAULT_SAVE_DIR = Path("activations/llava/distract_steer_new")

# DEFAULT_IMAGE_DIR = Path("libero_90_images")
# DEFAULT_SAVE_DIR = Path("activations/llava/distract_steer")


# def configure_hf_env() -> None:
#     os.environ.setdefault("HF_HOME", "/home/hice1/xyang645/scratch/hf_cache")
#     os.environ.setdefault("HF_HUB_CACHE", "/home/hice1/xyang645/scratch/hf_cache/hub")
#     os.environ.setdefault("TRANSFORMERS_CACHE", "/home/hice1/xyang645/scratch/hf_cache/transformers")
#     os.environ.setdefault("HF_DATASETS_CACHE", "/home/hice1/xyang645/scratch/hf_cache/datasets")


def resolve_image_dir(image_dir: Path) -> Path:
    if image_dir.exists():
        return image_dir
    fallback = Path("libero_90_images")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Image directory not found: {image_dir} or {fallback}")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def filename_to_prompt(image_path: Path) -> str:
    name = image_path.stem
    task_match = re.search(r"SCENE\d+_(.*?)_state", name)
    if not task_match:
        raise ValueError(f"Could not parse task from filename: {name}")
    task = task_match.group(1).replace("_", " ")
    return f"Task: {task}. Is the target object present in the scene? Answer YES or NO."


def filename_to_prompts_and_answers(image_path: Path) -> dict[str, str]:
    name = image_path.stem
    task_match = re.search(r"SCENE\d+_(.*?)_state", name)
    if not task_match:
        raise ValueError(f"Could not parse task from filename: {name}")

    task = task_match.group(1).replace("_", " ")
    no_part = name.split("_no_")[-1]
    no_part = re.sub(r"_\d+", "", no_part)
    missing = no_part.replace("_", " ")

    return {
        "yes": f"Task: {task}. Is the target object present in the scene? Answer YES or NO.",
        "no": f"Task: {task}. Is \"{missing}\" present in the scene? Answer YES or NO.",
    }


def build_image_splits(image_dir: Path, test_size: int = 60, seed: int = 12) -> dict[str, list[Path]]:
    all_images = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]
    )

    pair_map: dict[str, dict[str, Path]] = {}
    for p in all_images:
        if "clean" in p.stem:
            key = p.stem.split("_clean")[0]
            pair_map.setdefault(key, {})["clean"] = p
        elif "clutter" in p.stem:
            key = p.stem.split("_clutter")[0]
            pair_map.setdefault(key, {})["clutter"] = p

    pair_map = {k: v for k, v in pair_map.items() if "clean" in v and "clutter" in v}
    pair_keys = sorted(pair_map.keys())

    random.seed(seed)
    random.shuffle(pair_keys)

    test_keys = pair_keys[:test_size]
    train_keys = pair_keys[test_size:]

    return {
        "all": all_images,
        "clean_test": [pair_map[k]["clean"] for k in test_keys],
        "clutter_test": [pair_map[k]["clutter"] for k in test_keys],
        "clean_train": [pair_map[k]["clean"] for k in train_keys],
        "clutter_train": [pair_map[k]["clutter"] for k in train_keys],
    }


def load_model_processor(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = LlavaProcessor.from_pretrained(model_id, use_fast=False)
    _ = AutoTokenizer.from_pretrained(model_id, use_fast=False)

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
    return model, processor, device, dtype


def prepare_inputs(processor, model, dtype, prompt: str, image: Image.Image) -> dict[str, torch.Tensor]:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    return {
        k: (v.to(model.device, dtype=dtype) if v.dtype.is_floating_point else v.to(model.device))
        for k, v in inputs.items()
    }


def run_llava(model, processor, dtype, image_path: Path, task_prompt: str) -> str:
    prompt = f"USER: <image>\n{task_prompt}\nASSISTANT:"
    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, model, dtype, prompt, image)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_binary_prompts(model, processor, dtype, split_name: str, split_images: list[Path]) -> dict[str, float]:
    print(f"======================= {split_name.upper()} IMAGES ========================")
    tp = fp = fn = tn = 0

    for image_path in split_images:
        prompts = filename_to_prompts_and_answers(image_path)
        for expected in prompts:
            response = run_llava(model, processor, dtype, image_path, prompts[expected])
            predicted = response.strip().split()[0].lower() if response.strip() else ""
            print(f"Image: {image_path.name}")
            print(f"Prompt: {prompts[expected]}")
            print(f"Response: {response}\n")
            if expected == "yes":
                if predicted == expected:
                    tp += 1
                else:
                    fn += 1
                    print("---------------------false negative---------------------")
            else:
                if predicted == expected:
                    tn += 1
                else:
                    fp += 1
                    print("---------------------false positive---------------------")

    metrics = {
        "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
    }
    print(metrics)
    return metrics


def extract_diff_activations(
    model,
    processor,
    dtype,
    clean_images_train: list[Path],
    clutter_images_train: list[Path],
    save_dir: Path,
) -> Path:
    num_layers = model.language_model.config.num_hidden_layers
    layers = list(range(1, num_layers + 1))

    diff_attr_by_wrapper: dict[int, dict[int, list[torch.Tensor]]] = {}
    train_pairs = list(zip(clutter_images_train, clean_images_train))
    print("Train pairs:", len(train_pairs))

    amp_ctx = torch.cuda.amp.autocast() if model.device.type == "cuda" else contextlib.nullcontext()
    for index, (clutter_path, clean_path) in tqdm(enumerate(train_pairs), total=len(train_pairs)):
        task_prompt = filename_to_prompt(clean_path)
        query = f"USER: <image>\n{task_prompt}\nASSISTANT:"
        images = [
            Image.open(clutter_path).convert("RGB"),
            Image.open(clean_path).convert("RGB"),
        ]

        with torch.inference_mode(), amp_ctx:
            inputs = processor(text=[query] * 2, images=images, return_tensors="pt")
            inputs = {
                k: (v.to(model.device, dtype=dtype) if v.dtype.is_floating_point else v.to(model.device))
                for k, v in inputs.items()
            }
            output = model(**inputs, output_hidden_states=True)

        diff_attr_activations = {}
        for layer in layers:
            hidden_states = output.hidden_states[layer].detach().cpu()
            diff_attr_activations[layer] = hidden_states[0, -1] - hidden_states[1, -1]

        diff_attr_by_wrapper[index] = {layer: [] for layer in layers}
        for layer in layers:
            diff_attr_by_wrapper[index][layer].append(diff_attr_activations[layer])

    save_dir.mkdir(parents=True, exist_ok=True)
    diff_path = save_dir / "diff_activations_train.pt"
    torch.save(diff_attr_by_wrapper, diff_path)
    print("Saved diff activations to", diff_path)
    return diff_path


def extract_reference_activations(model, processor, dtype, all_images: list[Path], save_dir: Path) -> Path:
    num_layers = model.language_model.config.num_hidden_layers
    layers = list(range(1, num_layers + 1))

    ref_by_wrapper: dict[int, dict[int, list[torch.Tensor]]] = {}
    print("Reference images:", len(all_images))

    amp_ctx = torch.cuda.amp.autocast() if model.device.type == "cuda" else contextlib.nullcontext()
    for index, image_path in tqdm(enumerate(all_images), total=len(all_images)):
        reference_img = Image.open(image_path).convert("RGB")
        query = "USER: <image>\nWhat is the image about?\nASSISTANT:"

        with torch.inference_mode(), amp_ctx:
            inputs = processor(text=[query], images=[reference_img], return_tensors="pt")
            inputs = {
                k: (v.to(model.device, dtype=dtype) if v.dtype.is_floating_point else v.to(model.device))
                for k, v in inputs.items()
            }
            index_input_ids = inputs["input_ids"].shape[1]

            generate_ids = model.generate(
                **inputs,
                do_sample=True,
                max_length=256,
                temperature=0.2,
                top_p=0.9,
            )
            response = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=False,
            )

            inputs = processor(text=[query + response], images=[reference_img], return_tensors="pt")
            inputs = {
                k: (v.to(model.device, dtype=dtype) if v.dtype.is_floating_point else v.to(model.device))
                for k, v in inputs.items()
            }
            output = model(**inputs, output_hidden_states=True)

        img_activations = {}
        for layer in layers:
            hidden_states = output.hidden_states[layer].detach().cpu()
            img_activations[layer] = torch.mean(hidden_states[0, index_input_ids + 24 * 24 :], dim=0)

        ref_by_wrapper[index] = {layer: [] for layer in layers}
        for layer in layers:
            ref_by_wrapper[index][layer].append(img_activations[layer])

    save_dir.mkdir(parents=True, exist_ok=True)
    ref_path = save_dir / "reference_activations.pt"
    torch.save(ref_by_wrapper, ref_path)
    print("Saved reference activations to", ref_path)
    return ref_path


def load_activation_tensors(diff_path: Path, ref_path: Path):
    diff_by_wrapper = torch.load(diff_path, map_location="cpu")
    ref_by_wrapper = torch.load(ref_path, map_location="cpu")
    return diff_by_wrapper, ref_by_wrapper


def get_reference_activations_by_layer(layer: int, ref_by_wrapper, device) -> torch.Tensor:
    reference_activations = []
    for _, activations_per_layer in ref_by_wrapper.items():
        if layer in activations_per_layer:
            reference_activations.extend(activations_per_layer[layer])
    arr = np.array(reference_activations)
    arr = np.mean(arr, axis=0)
    return torch.from_numpy(arr).to(device)


def get_steer_activations_by_layer(layer: int, diff_by_wrapper, device) -> torch.Tensor:
    all_activations = []
    for _, activations_per_layer in diff_by_wrapper.items():
        if layer in activations_per_layer:
            all_activations.extend(activations_per_layer[layer])
    if not all_activations:
        raise ValueError(f"No activations found for layer {layer}")

    arr = np.array(all_activations)
    arr = np.mean(arr, axis=0)
    return torch.from_numpy(arr).to(device)


def get_prompt_text_positions(input_ids: torch.Tensor, processor, model) -> torch.Tensor:
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()

    if image_positions.numel() == 0:
        return torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)

    image_pos = int(image_positions[0].item())
    text_len = int(input_ids.shape[1])
    num_image_tokens = int(getattr(model.config, "image_seq_length", 24 * 24))
    shift = num_image_tokens - 1

    expanded_positions = []
    for pos in range(text_len):
        if pos == image_pos:
            continue
        expanded_positions.append(pos if pos < image_pos else pos + shift)

    return torch.tensor(expanded_positions, device=input_ids.device, dtype=torch.long)


def create_custom_forward_hook(
    steer_vector: torch.Tensor,
    reference_vector: torch.Tensor,
    steer_type: str,
    alpha: float,
    token_steer_mode: str,
    prompt_text_positions: torch.Tensor | None = None,
):
    def custom_forward_hook(_module, _input, output):
        hidden_states = output[0]

        if token_steer_mode == "last":
            r_feat = hidden_states[:, -1, :]
        elif token_steer_mode == "all_prompt_text":
            if hidden_states.shape[1] <= 1:
                return output
            if prompt_text_positions is None:
                return output

            valid_positions = prompt_text_positions[prompt_text_positions < hidden_states.shape[1]]
            if valid_positions.numel() == 0:
                return output
            r_feat = hidden_states[:, valid_positions, :]
        else:
            raise NotImplementedError(f"Unknown token_steer_mode: {token_steer_mode}")

        norm_steer_vector = torch.norm(steer_vector, p=2)
        unit_steer_vector = steer_vector / (norm_steer_vector + 1e-12)

        if steer_type == "linear":
            r_feat -= unit_steer_vector * alpha
        elif steer_type == "projection":
            project_feat = torch.sum((r_feat - reference_vector) * steer_vector, dim=-1)
            denom = torch.norm(r_feat - reference_vector, p=2, dim=-1) * torch.norm(steer_vector, p=2)
            project_feat = project_feat / (denom + 1e-12)
            clip_proj = torch.clamp(project_feat, min=0, max=1)
            coefficient = clip_proj * torch.norm(r_feat, p=2, dim=-1) * alpha
            r_feat -= coefficient.unsqueeze(-1) * unit_steer_vector
        elif steer_type != "no_steer":
            raise NotImplementedError(f"Unknown steer_type: {steer_type}")

        if token_steer_mode == "last":
            output[0][:, -1, :] = r_feat
        else:
            output[0][:, valid_positions, :] = r_feat
        return output

    return custom_forward_hook


def run_llava_steered(
    model,
    processor,
    dtype,
    image_path: Path,
    task_prompt: str,
    steer_type: str,
    alpha: float,
    layer: int,
    token_steer_mode: str,
    diff_by_wrapper,
    ref_by_wrapper,
) -> str:
    prompt = f"USER: <image>\n{task_prompt}\nASSISTANT:"
    image = Image.open(image_path).convert("RGB")
    inputs = prepare_inputs(processor, model, dtype, prompt, image)

    steer_activations = get_steer_activations_by_layer(layer, diff_by_wrapper, model.device)
    reference_activations = get_reference_activations_by_layer(layer, ref_by_wrapper, model.device)
    prompt_text_positions = None
    if token_steer_mode == "all_prompt_text":
        prompt_text_positions = get_prompt_text_positions(inputs["input_ids"], processor, model)
    custom_hook = create_custom_forward_hook(
        steer_activations,
        reference_activations,
        steer_type,
        alpha,
        token_steer_mode,
        prompt_text_positions,
    )
    hook = model.language_model.base_model.layers[layer - 1].register_forward_hook(custom_hook)

    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=10)
    finally:
        hook.remove()

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_steered_split(
    model,
    processor,
    dtype,
    split_name: str,
    split_images: list[Path],
    layer: int,
    alpha: float,
    steer_type: str,
    token_steer_mode: str,
    diff_by_wrapper,
    ref_by_wrapper,
) -> dict[str, dict[str, float]]:
    print(f"======================= {split_name.upper()} IMAGES ========================")
    out: dict[str, dict[str, float]] = {}

    if steer_type == "all":
        steer_configs = [("no_steer", 0.0), ("linear", alpha), ("projection", alpha)]
    elif steer_type == "no_steer":
        steer_configs = [("no_steer", 0.0)]
    else:
        steer_configs = [(steer_type, alpha)]

    for curr_steer_type, steer_alpha in steer_configs:
        print(f"============ {curr_steer_type} ============")
        tp = fp = fn = tn = 0

        for image_ind, image_path in enumerate(split_images):
            print(f"{image_ind + 1}/{len(split_images)}")
            prompts = filename_to_prompts_and_answers(image_path)
            for expected in prompts:
                response = run_llava_steered(
                    model,
                    processor,
                    dtype,
                    image_path,
                    prompts[expected],
                    curr_steer_type,
                    steer_alpha,
                    layer,
                    token_steer_mode,
                    diff_by_wrapper,
                    ref_by_wrapper,
                )
                predicted = response.strip().split()[0].lower() if response.strip() else ""
                print(f"Image: {image_path.name}")
                print(f"Prompt: {prompts[expected]}")
                print(f"Response: {response}\n")

                if expected == "yes":
                    if predicted == expected:
                        tp += 1
                    else:
                        fn += 1
                        print("---------------------false negative---------------------")
                else:
                    if predicted == expected:
                        tn += 1
                    else:
                        fp += 1
                        print("---------------------false positive---------------------")

        metrics = {
            "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
        }
        print(metrics)
        out[curr_steer_type] = metrics

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distract steer all-token pipeline.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--layer", type=int, default=16, help="1-based decoder layer index")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--steer-type",
        choices=["all", "no_steer", "linear", "projection"],
        default="all",
        help="Steering mode(s) to run during steer_eval.",
    )
    parser.add_argument(
        "--token-steer-mode",
        choices=["last", "all_prompt_text"],
        default="last",
        help="Where to apply steering within decoder hidden states.",
    )

    parser.add_argument(
        "--stage",
        choices=["baseline_eval", "extract_diff", "extract_ref", "steer_eval", "all"],
        default="all",
    )
    parser.add_argument(
        "--split",
        choices=["clean", "clutter", "both"],
        default="both",
        help="Which test split(s) to evaluate for eval stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # configure_hf_env()

    image_dir = resolve_image_dir(args.image_dir)
    splits = build_image_splits(image_dir, test_size=args.test_size, seed=args.seed)

    print(
        f"Found {len(splits['clean_test']) + len(splits['clean_train'])} clean images and "
        f"{len(splits['clutter_test']) + len(splits['clutter_train'])} clutter images."
    )

    model, processor, _, dtype = load_model_processor(args.model_id)

    eval_splits = []
    if args.split in {"clean", "both"}:
        eval_splits.append(("clean", splits["clean_test"]))
    if args.split in {"clutter", "both"}:
        eval_splits.append(("clutter", splits["clutter_test"]))

    if args.stage in {"baseline_eval", "all"}:
        for split_name, split_images in eval_splits:
            evaluate_binary_prompts(model, processor, dtype, split_name, split_images)

    if args.stage in {"extract_diff", "all"}:
        extract_diff_activations(
            model,
            processor,
            dtype,
            splits["clean_train"],
            splits["clutter_train"],
            args.save_dir,
        )

    if args.stage in {"extract_ref", "all"}:
        extract_reference_activations(model, processor, dtype, splits["all"], args.save_dir)

    if args.stage in {"steer_eval", "all"}:
        diff_path = args.save_dir / "diff_activations_train.pt"
        ref_path = args.save_dir / "reference_activations.pt"
        if not diff_path.exists() or not ref_path.exists():
            raise FileNotFoundError(
                f"Missing activation files for steer_eval: {diff_path} and/or {ref_path}. "
                "Run --stage extract_diff and --stage extract_ref first."
            )

        diff_by_wrapper, ref_by_wrapper = load_activation_tensors(diff_path, ref_path)
        for split_name, split_images in eval_splits:
            evaluate_steered_split(
                model,
                processor,
                dtype,
                split_name,
                split_images,
                args.layer,
                args.alpha,
                args.steer_type,
                args.token_steer_mode,
                diff_by_wrapper,
                ref_by_wrapper,
            )


if __name__ == "__main__":
    main()
