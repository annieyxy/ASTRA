# import numpy as np
# import pandas as pd
# import torch as ch
# from numpy.typing import NDArray
# from typing import Dict, Any, Optional, List
# import logging
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from .context_partitioner import BaseContextPartitioner, SimpleContextPartitioner, SimpleImagePartitioner
# from .solver import BaseSolver, LassoRegression
# from .utils import (
#     get_masks_and_logit_probs,
#     get_llava_masks_and_logit_probs,
#     get_vlm_masks_and_logit_probs,
#     aggregate_logit_probs,
#     split_text,
#     highlight_word_indices,
#     get_attributions_df,
#     get_vlm_attributions_df,
#     char_to_token,
# )

# from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration
# from PIL import Image

# import argparse
# import os
# import random

# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# from PIL import Image
# from torchvision.utils import save_image

# class ImageAttr_LLAVA:
#     def __init__(
#         self,
#         model: Any,
#         tokenizer: Any,
#         pretrained_model_name_or_path: str,
#         processor: None
#     ) -> None:

#         self.model_name = pretrained_model_name_or_path
#         self.model = model
#         self.tokenizer = tokenizer
#         self.processor = processor

#         self.logger = logging.getLogger("ImageAttr")
#         self.logger.setLevel(logging.DEBUG)  # TODO: change to INFO later
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
    
#     @classmethod
#     def from_pretrained(
#         cls,
#         pretrained_model_name_or_path: str=" ",
#         tokenizer_name: str = " ",
#         device: str = "cuda",
#         model_kwargs: Dict[str, Any] = {},
#         tokenizer_kwargs: Dict[str, Any] = {},
#         **kwargs: Dict[str, Any],
#     ) -> "ImageAttr_LLAVA":

#         model_name = 'llava-hf/llava-1.5-13b-hf'
#         image_processor = LlavaProcessor.from_pretrained(model_name) # liuhaotian/llava-v1.6-vicuna-7b
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = LlavaForConditionalGeneration.from_pretrained(
#                     model_name, torch_dtype=ch.float16, device_map='auto')
#         return cls(model, tokenizer, model_name, image_processor, **kwargs)

#     def init_image_query(
#         self,
#         image: Image,
#         query: str,
#         source_type: str = "patch",
#         generate_kwargs: Optional[Dict[str, Any]] = None,
#         num_ablations: int = 64,
#         ablation_keep_prob: float = 0.5,
#         batch_size: int = 1,
#         solver: Optional[BaseSolver] = None,
#     ):
#         assert source_type=='patch'
#         self.partitioner = SimpleImagePartitioner(image, source_type=source_type, image_size=336)
   
#         self.query = 'USER: <image>\n'+query+'\nASSISTANT:'
#         self.num_ablations = num_ablations
#         self.ablation_keep_prob = ablation_keep_prob
#         self.batch_size = batch_size
#         self.solver = solver or LassoRegression()
#         self._cache = {}

#     def _get_query_ids(
#         self,
#         return_prompt: bool = False,
#     ):
#         chat_prompt = self.query
#         chat_prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)

#         if return_prompt:
#             return chat_prompt_ids, chat_prompt
#         else:
#             return chat_prompt_ids

#     @property
#     def _response_start(self):
#         prompt_ids = self._get_query_ids()
#         return len(prompt_ids)
    
#     @property
#     def _output(self):
#         if self._cache.get("output") is None:
#             prompt_ids, prompt = self._get_query_ids(return_prompt=True)
#             inputs = self.processor(text=prompt, images=self.partitioner.get_image(), return_tensors="pt").to("cuda", torch.float16)
#             generate_ids = self.model.generate(**inputs, 
#                             do_sample=True, max_length=1024,
#                             temperature=0.2, top_p=0.9,
#                         )
#             response = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            
#             self._cache["output"] = prompt + response
#         return self._cache["output"]
    
#     def specify_output(self, jb_output):
#         if self._cache.get("output") is None:
#             prompt_ids, prompt = self._get_query_ids(return_prompt=True)
#             self._cache["output"] = prompt + jb_output
#         return self._cache["output"]
    
#     @property
#     def _output_tokens(self):
#         return self.tokenizer(self._output, add_special_tokens=False)

#     @property
#     def _response_ids(self):
#         return self._output_tokens["input_ids"][self._response_start :]

#     @property
#     def response(self): 
#         output_tokens = self._output_tokens
#         char_response_start = output_tokens.token_to_chars(self._response_start).start
#         return self._output[char_response_start:]

#     @property
#     def num_sources(self) -> int:
#         """
#         The number of sources within the context. I.e., the number of sources
#         that the context is partitioned into.

#         Returns:
#             int:
#                 The number of sources in the context.
#         """
#         return self.partitioner.num_sources

#     @property
#     def sources(self) -> List[np.array]:
#         """
#         The sources within the context. I.e., the context as a list
#         where each element is a source.

#         Returns:
#             List[str]:
#                 The sources within the context.
#         """
#         return self.partitioner.sources

#     # have the index problem to fix
#     def _char_range_to_token_range(self, start_index, end_index):
#         output_tokens = self._output_tokens
#         response_start = self._response_start
#         offset = output_tokens.token_to_chars(response_start).start
#         ids_start_index = char_to_token(output_tokens, start_index + offset)
#         ids_end_index = char_to_token(output_tokens, end_index + offset - 1) + 1
#         return ids_start_index - response_start, ids_end_index - response_start

#     def _indices_to_token_indices(self, start_index=None, end_index=None):
#         if start_index is None or end_index is None:
#             start_index = 0
#             end_index = len(self.response)
#         if not (0 <= start_index < end_index <= len(self.response)):
#             raise ValueError(
#                 f"Invalid selection range ({start_index}, {end_index}). "
#                 f"Please select any range within (0, {len(self.response)})."
#             )
#         return self._char_range_to_token_range(start_index, end_index)

#     def _compute_vlm_masks_and_logit_probs(self) -> None:
#         self._cache["reg_masks"], self._cache["reg_logit_probs"] = (
#             get_llava_masks_and_logit_probs(
#                 self.model,
#                 self.processor,
#                 self.tokenizer,
#                 self.num_ablations,
#                 self.num_sources,
#                 self.partitioner,
#                 self._get_query_ids,
#                 self.response,
#                 self._response_ids,
#                 self.ablation_keep_prob,
#                 self.batch_size,
#             )
#         )

#     @property
#     def _masks(self):
#         if self._cache.get("reg_masks") is None:
#             self._compute_vlm_masks_and_logit_probs()
#         return self._cache["reg_masks"]

#     @property
#     def _logit_probs(self):
#         if self._cache.get("reg_logit_probs") is None:
#             self._compute_vlm_masks_and_logit_probs()
#         return self._cache["reg_logit_probs"]

#     def _get_attributions_for_ids_range(self, ids_start_idx, ids_end_idx) -> tuple:
#         outputs = aggregate_logit_probs(self._logit_probs[:, ids_start_idx:ids_end_idx])
#         num_output_tokens = ids_end_idx - ids_start_idx
        
#         weight, bias = self.solver.fit(self._masks, outputs, num_output_tokens)
#         return weight, bias

#     def get_attributions(
#         self,
#         start_idx: Optional[int] = None,
#         end_idx: Optional[int] = None,
#         as_dataframe: bool = False,
#         top_k: Optional[int] = None,
#         verbose: bool = True,
#     ) -> NDArray | Any:
#         """
#         Get the attributions for (part of) the response.

#         Arguments:
#             start_idx (int, optional):
#                 Start index of the part to attribute to. If None, defaults to
#                 the start of the response.
#             end_idx (int, optional):
#                 End index of the part to attribute to. If None, defaults to the
#                 end of the response.
#             as_dataframe (bool, optional):
#                 If True, return the attributions as a stylized dataframe in
#                 sorted order. Otherwise, return them as a numpy array where
#                 the ith element corresponds to the score of the ith source
#                 within the context. Defaults to False.
#             top_k (int, optional):
#                 Only used if as_dataframe is True. Number of top attributions to
#                 return. If None, all attributions are returned. Defaults to None.
#             verbose (bool, optional):
#                 If True, print the selected part of the response. Defaults to
#                 True.

#         Returns:
#             NDArray | Any:
#                 If as_dataframe is False, return a numpy array where the ith element
#                 corresponds to the score of the ith source within the context.
#                 Otherwise, return a stylized dataframe in sorted order.
#         """
#         if self.num_sources == 0:
#             self.logger.warning("No sources to attribute to!")
#             return np.array([])

#         if not as_dataframe and top_k is not None:
#             self.logger.warning("top_k is ignored when not using dataframes.")

#         ids_start_idx, ids_end_idx = self._indices_to_token_indices(start_idx, end_idx)
#         selected_text = self.response[start_idx:end_idx]
#         selected_tokens = self._response_ids[ids_start_idx:ids_end_idx]
#         decoded_text = self.tokenizer.decode(selected_tokens)
#         if selected_text.strip() not in decoded_text.strip() and verbose:
#             self.logger.warning(
#                 "Decoded selected tokens do not match selected text.\n"
#                 "If the following look close enough, feel free to ignore:\n"
#                 "What you selected: %s\nWhat is being attributed: %s",
#                 selected_text.strip(),
#                 decoded_text.strip(),
#             )

#         if verbose:
#             print(f"Attributed: {decoded_text.strip()}")

#         # _bias is the bias term in the l1 regression
#         attributions, _bias = self._get_attributions_for_ids_range(
#             ids_start_idx,
#             ids_end_idx,
#         )

#         if as_dataframe:
#             return get_vlm_attributions_df(attributions, self.partitioner, top_k=top_k)
#         else:
#             return attributions
        
        

import numpy as np
import pandas as pd
import torch as ch
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from .context_partitioner import BaseContextPartitioner, SimpleContextPartitioner, SimpleImagePartitioner
from .solver import BaseSolver, LassoRegression
from .utils import (
    get_masks_and_logit_probs,
    get_llava_masks_and_logit_probs,
    get_vlm_masks_and_logit_probs,
    aggregate_logit_probs,
    split_text,
    highlight_word_indices,
    get_attributions_df,
    get_vlm_attributions_df,
    char_to_token,
)

from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration
try:
    from transformers import AutoImageProcessor
except ImportError:  # older transformers
    AutoImageProcessor = None
try:
    from transformers import CLIPImageProcessor
except ImportError:  # older transformers
    CLIPImageProcessor = None
from PIL import Image

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image

class ImageAttr_LLAVA:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        pretrained_model_name_or_path: str,
        processor: None
    ) -> None:

        self.model_name = pretrained_model_name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        self.logger = logging.getLogger("ImageAttr")
        self.logger.setLevel(logging.DEBUG)  # TODO: change to INFO later
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str=" ",
        tokenizer_name: str = " ",
        device: str = "cuda",
        model_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        **kwargs: Dict[str, Any],
    ) -> "ImageAttr_LLAVA":

        model_name = 'llava-hf/llava-1.5-13b-hf'
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        try:
            image_processor = LlavaProcessor.from_pretrained(model_name, use_fast=False)
        except Exception:
            # Fallback for processor/tokenizer issues (e.g., bad cached fast tokenizer file).
            if AutoImageProcessor is not None:
                image_processor = AutoImageProcessor.from_pretrained(model_name)
            elif CLIPImageProcessor is not None:
                image_processor = CLIPImageProcessor.from_pretrained(model_name)
            else:
                raise
            image_processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = LlavaForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=ch.float16, device_map='auto')
        return cls(model, tokenizer, model_name, image_processor, **kwargs)

    def init_image_query(
        self,
        image: Image,
        query: str,
        source_type: str = "patch",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_ablations: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: Optional[BaseSolver] = None,
    ):
        assert source_type=='patch'
        self.partitioner = SimpleImagePartitioner(image, source_type=source_type, image_size=336)
   
        self.query = 'USER: <image>\n'+query+'\nASSISTANT:'
        self.num_ablations = num_ablations
        self.ablation_keep_prob = ablation_keep_prob
        self.batch_size = batch_size
        self.solver = solver or LassoRegression()
        self._cache = {}

    def _get_query_ids(
        self,
        return_prompt: bool = False,
    ):
        chat_prompt = self.query
        chat_prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)

        if return_prompt:
            return chat_prompt_ids, chat_prompt
        else:
            return chat_prompt_ids

    @property
    def _response_start(self):
        prompt_ids = self._get_query_ids()
        return len(prompt_ids)
    
    @property
    def _output(self):
        if self._cache.get("output") is None:
            prompt_ids, prompt = self._get_query_ids(return_prompt=True)
            inputs = self.processor(text=prompt, images=self.partitioner.get_image(), return_tensors="pt").to("cuda", torch.float16)
            generate_ids = self.model.generate(**inputs, 
                            do_sample=True, max_length=1024,
                            temperature=0.2, top_p=0.9,
                        )
            response = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            
            self._cache["output"] = prompt + response
        return self._cache["output"]
    
    def specify_output(self, jb_output):
        if self._cache.get("output") is None:
            prompt_ids, prompt = self._get_query_ids(return_prompt=True)
            self._cache["output"] = prompt + jb_output
        return self._cache["output"]
    
    @property
    def _output_tokens(self):
        return self.tokenizer(self._output, add_special_tokens=False)

    @property
    def _response_ids(self):
        return self._output_tokens["input_ids"][self._response_start :]

    @property
    def response(self): 
        # Avoid token_to_chars for slow tokenizers by slicing on the prompt text.
        _, prompt = self._get_query_ids(return_prompt=True)
        return self._output[len(prompt):]

    @property
    def num_sources(self) -> int:
        """
        The number of sources within the context. I.e., the number of sources
        that the context is partitioned into.

        Returns:
            int:
                The number of sources in the context.
        """
        return self.partitioner.num_sources

    @property
    def sources(self) -> List[np.array]:
        """
        The sources within the context. I.e., the context as a list
        where each element is a source.

        Returns:
            List[str]:
                The sources within the context.
        """
        return self.partitioner.sources

    # have the index problem to fix
    def _char_range_to_token_range(self, start_index, end_index):
        output_tokens = self._output_tokens
        response_start = self._response_start
        try:
            offset = output_tokens.token_to_chars(response_start).start
        except Exception:
            # Slow tokenizer fallback: derive token indices from response substrings.
            response_text = self.response
            ids_start_index = len(self.tokenizer.encode(response_text[:start_index], add_special_tokens=False))
            ids_end_index = len(self.tokenizer.encode(response_text[:end_index], add_special_tokens=False))
            return ids_start_index, ids_end_index
        ids_start_index = char_to_token(output_tokens, start_index + offset)
        ids_end_index = char_to_token(output_tokens, end_index + offset - 1) + 1
        return ids_start_index - response_start, ids_end_index - response_start

    def _indices_to_token_indices(self, start_index=None, end_index=None):
        if start_index is None or end_index is None:
            start_index = 0
            end_index = len(self.response)
        if not (0 <= start_index < end_index <= len(self.response)):
            raise ValueError(
                f"Invalid selection range ({start_index}, {end_index}). "
                f"Please select any range within (0, {len(self.response)})."
            )
        return self._char_range_to_token_range(start_index, end_index)

    def _compute_vlm_masks_and_logit_probs(self) -> None:
        self._cache["reg_masks"], self._cache["reg_logit_probs"] = (
            get_llava_masks_and_logit_probs(
                self.model,
                self.processor,
                self.tokenizer,
                self.num_ablations,
                self.num_sources,
                self.partitioner,
                self._get_query_ids,
                self.response,
                self._response_ids,
                self.ablation_keep_prob,
                self.batch_size,
            )
        )

    @property
    def _masks(self):
        if self._cache.get("reg_masks") is None:
            self._compute_vlm_masks_and_logit_probs()
        return self._cache["reg_masks"]

    @property
    def _logit_probs(self):
        if self._cache.get("reg_logit_probs") is None:
            self._compute_vlm_masks_and_logit_probs()
        return self._cache["reg_logit_probs"]

    def _get_attributions_for_ids_range(self, ids_start_idx, ids_end_idx) -> tuple:
        outputs = aggregate_logit_probs(self._logit_probs[:, ids_start_idx:ids_end_idx])
        num_output_tokens = ids_end_idx - ids_start_idx
        
        weight, bias = self.solver.fit(self._masks, outputs, num_output_tokens)
        return weight, bias

    def get_attributions(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        as_dataframe: bool = False,
        top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> NDArray | Any:
        """
        Get the attributions for (part of) the response.

        Arguments:
            start_idx (int, optional):
                Start index of the part to attribute to. If None, defaults to
                the start of the response.
            end_idx (int, optional):
                End index of the part to attribute to. If None, defaults to the
                end of the response.
            as_dataframe (bool, optional):
                If True, return the attributions as a stylized dataframe in
                sorted order. Otherwise, return them as a numpy array where
                the ith element corresponds to the score of the ith source
                within the context. Defaults to False.
            top_k (int, optional):
                Only used if as_dataframe is True. Number of top attributions to
                return. If None, all attributions are returned. Defaults to None.
            verbose (bool, optional):
                If True, print the selected part of the response. Defaults to
                True.

        Returns:
            NDArray | Any:
                If as_dataframe is False, return a numpy array where the ith element
                corresponds to the score of the ith source within the context.
                Otherwise, return a stylized dataframe in sorted order.
        """
        if self.num_sources == 0:
            self.logger.warning("No sources to attribute to!")
            return np.array([])

        if not as_dataframe and top_k is not None:
            self.logger.warning("top_k is ignored when not using dataframes.")

        ids_start_idx, ids_end_idx = self._indices_to_token_indices(start_idx, end_idx)
        selected_text = self.response[start_idx:end_idx]
        selected_tokens = self._response_ids[ids_start_idx:ids_end_idx]
        decoded_text = self.tokenizer.decode(selected_tokens)
        if selected_text.strip() not in decoded_text.strip() and verbose:
            self.logger.warning(
                "Decoded selected tokens do not match selected text.\n"
                "If the following look close enough, feel free to ignore:\n"
                "What you selected: %s\nWhat is being attributed: %s",
                selected_text.strip(),
                decoded_text.strip(),
            )

        if verbose:
            print(f"Attributed: {decoded_text.strip()}")

        # _bias is the bias term in the l1 regression
        attributions, _bias = self._get_attributions_for_ids_range(
            ids_start_idx,
            ids_end_idx,
        )

        if as_dataframe:
            return get_vlm_attributions_df(attributions, self.partitioner, top_k=top_k)
        else:
            return attributions
        
        
