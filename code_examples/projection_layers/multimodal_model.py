"""Multimodal model with projection layers."""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel

from config import Config
from projection_layers import create_projection


class MultiModalModel(nn.Module):
    """Multimodal model with a vision encoder, projection, and LLM."""

    def __init__(self, config: Config, projection_type: str = "mlp"):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.image_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        vision_dim = self.vision_model.config.hidden_size
        llm_dim = self.llm.config.hidden_size
        self.projection = create_projection(projection_type, vision_dim, llm_dim)

    def _build_chat_input_ids(self, messages, add_generation_prompt: bool, device: str) -> torch.Tensor:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        return encoded.input_ids.to(device)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        visual_tokens = vision_outputs.last_hidden_state
        projected = self.projection(visual_tokens)
        num_visual = projected.size(1)

        text_embeds = self.llm.model.embed_tokens(input_ids)
        combined_embeds = torch.cat([projected, text_embeds], dim=1)

        visual_mask = torch.ones(projected.size(0), num_visual, dtype=attention_mask.dtype, device=attention_mask.device)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        combined_labels = None
        if labels is not None:
            visual_labels = torch.full((labels.size(0), num_visual), -100, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([visual_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            return_dict=True,
        )

    @staticmethod
    def _load_image(image_path: Union[str, Path]) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _encode_image(self, image: Image.Image, device: str) -> torch.Tensor:
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(device)

    @torch.no_grad()
    def generate_from_pixel_values(
        self,
        pixel_values: torch.Tensor,
        device: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Describe this image.",
    ) -> List[str]:
        self.eval()
        batch_size = pixel_values.size(0)

        # Get the dtype of the LLM to ensure all tensors match
        llm_dtype = next(self.llm.parameters()).dtype

        vision_outputs = self.vision_model(pixel_values=pixel_values)
        visual_tokens = vision_outputs.last_hidden_state
        projected = self.projection(visual_tokens)

        # Convert to LLM dtype
        projected = projected.to(llm_dtype)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_ids = self._build_chat_input_ids(messages, add_generation_prompt=True, device=device)
        if batch_size > 1:
            input_ids = prompt_ids.expand(batch_size, -1)
        else:
            input_ids = prompt_ids

        text_embeds = self.llm.model.embed_tokens(input_ids)
        # Convert to LLM dtype
        text_embeds = text_embeds.to(llm_dtype)

        combined_embeds = torch.cat([projected, text_embeds], dim=1)

        visual_mask = torch.ones(batch_size, projected.size(1), dtype=torch.long, device=device)
        text_mask = torch.ones(batch_size, input_ids.size(1), dtype=torch.long, device=device)
        combined_mask = torch.cat([visual_mask, text_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        captions = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            if "assistant" in text:
                text = text.split("assistant")[-1].strip()
            captions.append(text.strip())
        return captions

    @torch.no_grad()
    def generate_caption(
        self,
        image_path: Union[str, Path],
        device: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        user_prompt: str = "Describe this image.",
    ) -> str:
        image = self._load_image(image_path)
        pixel_values = self._encode_image(image, device)
        captions = self.generate_from_pixel_values(
            pixel_values,
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return captions[0] if captions else ""

    @torch.no_grad()
    def generate_caption_from_image(
        self,
        image: Image.Image,
        device: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        user_prompt: str = "Describe this image.",
    ) -> str:
        pixel_values = self._encode_image(image, device)
        captions = self.generate_from_pixel_values(
            pixel_values,
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return captions[0] if captions else ""
