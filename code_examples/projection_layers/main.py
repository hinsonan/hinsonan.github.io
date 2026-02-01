from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import time
import concurrent.futures
from collections import OrderedDict
from PIL import Image
import requests
from io import BytesIO

class ImagePrefetchCache:
    """
    Async prefetch cache for remote images. Downloads are submitted to a thread
    pool eagerly; get() blocks only if the target download hasn't finished yet.
    Failed downloads are not cached — get() raises, letting the caller retry.
    """
    def __init__(self, max_workers=4, max_cached=64):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures = OrderedDict()  # idx -> Future[Image], insertion-ordered
        self._max_cached = max_cached

    def prefetch(self, idx, url):
        """Submit a download for idx if not already in-flight."""
        if idx in self._futures:
            return
        while len(self._futures) >= self._max_cached:
            _, evicted = self._futures.popitem(last=False)
            evicted.cancel()
        self._futures[idx] = self._executor.submit(self._download, url)

    def get(self, idx, url, timeout=30):
        """
        Retrieve image for idx. Submits the download if not yet prefetched.
        Raises on download failure or timeout.
        """
        self.prefetch(idx, url)
        future = self._futures.pop(idx)
        return future.result(timeout=timeout)

    @staticmethod
    def _download(url):
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")


class CocoCaptionsDataset(Dataset):
    def __init__(self, tokenizer, image_processor, split="train", max_length=512,
                 system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                 prefetch_size=16, prefetch_workers=4):
        """
        Dataset for COCO Captions formatted for Qwen2.5 chat template.

        Args:
            tokenizer: Qwen tokenizer with chat template
            image_processor: Image processor for vision encoder
            split: Dataset split ("train" or "validation")
            max_length: Maximum sequence length for tokenization
            system_prompt: System prompt for the chat template
            prefetch_size: Number of images to prefetch ahead
            prefetch_workers: Number of background download threads
        """
        self.dataset = load_dataset("yerevann/coco-karpathy", split=split)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.prefetch_size = prefetch_size
        self._prefetch_cache = ImagePrefetchCache(
            max_workers=prefetch_workers,
            max_cached=prefetch_size * 2
        )
        self._max_retries = 5

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, idx):
        """
        Load image via the prefetch cache. On failure, retry with subsequent
        indices instead of substituting fabricated data. Returns (image, item_idx).
        """
        # Speculatively prefetch upcoming indices
        for offset in range(1, self.prefetch_size + 1):
            next_idx = (idx + offset) % len(self)
            self._prefetch_cache.prefetch(next_idx, self.dataset[next_idx]["url"])

        current_idx = idx
        for attempt in range(self._max_retries):
            try:
                url = self.dataset[current_idx]["url"]
                image = self._prefetch_cache.get(current_idx, url)
                return image, current_idx
            except Exception as e:
                logger.warning(
                    f"Failed to load image at index {current_idx} "
                    f"(attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                current_idx = (current_idx + 1) % len(self)

        raise RuntimeError(
            f"Failed to load any valid image after {self._max_retries} retries "
            f"starting from index {idx}"
        )

    def __getitem__(self, idx):
        image, item_idx = self._load_image(idx)
        item = self.dataset[item_idx]

        # Deterministic caption selection: reproducible given the same item index
        captions = item["sentences"]
        caption = captions[item_idx % len(captions)]

        # Process image for vision encoder
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # Format caption using Qwen chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Describe this image."},
            {"role": "assistant", "content": caption}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Shared kwargs ensure prompt-length calculation uses identical settings
        # as the full-sequence tokenization (critical for correct label masking)
        tokenize_kwargs = dict(max_length=self.max_length, padding=False, truncation=True)

        encoding = self.tokenizer(text, return_tensors="pt", **tokenize_kwargs)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask prompt tokens so loss is computed only on the assistant's caption.
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        prompt_length = len(self.tokenizer(prompt_text, **tokenize_kwargs)["input_ids"])
        labels[:prompt_length] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption
        }

def collate_fn(batch, tokenizer):
    """
    Custom collate function to dynamically pad sequences to the longest in the batch.
    This reduces unnecessary padding tokens.
    """
    # Extract individual components
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    captions = [item["caption"] for item in batch]

    # Find max length in this batch
    max_len = max(item["input_ids"].size(0) for item in batch)

    # Pad sequences to max_len in this batch
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        padding_len = max_len - seq_len

        # Pad input_ids with pad_token_id
        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
        ])

        # Pad attention_mask with 0s
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(padding_len, dtype=torch.long)
        ])

        # Pad labels with -100 (ignore index)
        padded_labels = torch.cat([
            item["labels"],
            torch.full((padding_len,), -100, dtype=torch.long)
        ])

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
        labels_list.append(padded_labels)

    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "caption": captions
    }

def create_dataloader(tokenizer, image_processor, split="train", batch_size=8, shuffle=None, num_workers=2, max_length=512,
                      system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                      prefetch_size=16, prefetch_workers=4):
    """
    Create a DataLoader for COCO captions formatted for multimodal training.

    Args:
        tokenizer: Qwen tokenizer with chat template
        image_processor: Image processor for vision encoder
        split: Dataset split ("train" or "validation")
        batch_size: Batch size
        shuffle: Whether to shuffle (defaults to True for train, False otherwise)
        num_workers: Number of worker processes for data loading
        max_length: Maximum sequence length
        system_prompt: System prompt for the chat template
        prefetch_size: Number of images to prefetch ahead per worker
        prefetch_workers: Number of background download threads per worker

    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = CocoCaptionsDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        split=split,
        max_length=max_length,
        system_prompt=system_prompt,
        prefetch_size=prefetch_size,
        prefetch_workers=prefetch_workers
    )

    # Use lambda to pass tokenizer to collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Speed up GPU transfer
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

# === Projection Layers ===

class MLPProjection(torch.nn.Module):
    """Two-layer MLP projection (LLaVA-1.5 style).
    Linear(vision_dim → llm_dim) → GELU → Linear(llm_dim → llm_dim).
    Preserves all patch tokens. Both linear layers use llm_dim — there is
    no expanded intermediate dimension.

    Paper: https://arxiv.org/abs/2310.03744 (Liu et al., 2023 - Improved Baselines with Visual Instruction Tuning)
    """
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, llm_dim),
            torch.nn.GELU(),
            torch.nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # (batch, num_patches, vision_dim) → (batch, num_patches, llm_dim)
        return self.net(visual_tokens)


class QFormerProjection(torch.nn.Module):
    """Q-Former projection (BLIP-2 style).
    Learnable query tokens self-attend and cross-attend to visual tokens.
    Cross-attention is inserted every `cross_attn_every` layers — BLIP-2
    defaults to every other layer, not every layer.

    Paper: https://arxiv.org/abs/2301.12597 (Li et al., 2023 - BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models)
    """
    def __init__(self, vision_dim: int, llm_dim: int,
                 num_queries: int = 32, num_layers: int = 12,
                 num_heads: int = 12, cross_attn_every: int = 2):
        super().__init__()
        self.query_tokens = torch.nn.Parameter(torch.randn(1, num_queries, vision_dim))

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = torch.nn.ModuleDict({
                "self_attn": torch.nn.MultiheadAttention(
                    vision_dim, num_heads, batch_first=True),
                "self_norm": torch.nn.LayerNorm(vision_dim),
                "ffn": torch.nn.Sequential(
                    torch.nn.Linear(vision_dim, vision_dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(vision_dim * 4, vision_dim),
                ),
                "ffn_norm": torch.nn.LayerNorm(vision_dim),
            })
            # Cross-attention only on designated layers
            if i % cross_attn_every == 0:
                layer["cross_attn"] = torch.nn.MultiheadAttention(
                    vision_dim, num_heads, batch_first=True)
                layer["cross_norm"] = torch.nn.LayerNorm(vision_dim)
            self.layers.append(layer)

        self.output_proj = torch.nn.Linear(vision_dim, llm_dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # (batch, num_patches, vision_dim) → (batch, num_queries, llm_dim)
        batch_size = visual_tokens.shape[0]
        queries = self.query_tokens.expand(batch_size, -1, -1)

        for layer in self.layers:
            sa_out, _ = layer["self_attn"](queries, queries, queries)
            queries = layer["self_norm"](queries + sa_out)

            if "cross_attn" in layer:
                ca_out, _ = layer["cross_attn"](queries, visual_tokens, visual_tokens)
                queries = layer["cross_norm"](queries + ca_out)

            queries = layer["ffn_norm"](queries + layer["ffn"](queries))

        return self.output_proj(queries)


class PerceiverResamplerProjection(torch.nn.Module):
    """Perceiver Resampler projection (Flamingo style).
    K and V are derived from cat(visual_tokens, latents), so latents implicitly
    attend to each other within the same cross-attention op — no separate
    self-attention block. Pre-norm throughout, matching the Flamingo source.

    Paper: https://arxiv.org/abs/2204.14198 (Alayrac et al., 2022 - Flamingo: a Visual Language Model for Few-Shot Learning)
    """
    def __init__(self, vision_dim: int, llm_dim: int,
                 num_latents: int = 64, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.latents = torch.nn.Parameter(torch.randn(1, num_latents, vision_dim))

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(torch.nn.ModuleDict({
                "norm_media":   torch.nn.LayerNorm(vision_dim),
                "norm_latents": torch.nn.LayerNorm(vision_dim),
                "attn":         torch.nn.MultiheadAttention(
                    vision_dim, num_heads, batch_first=True),
                "ffn": torch.nn.Sequential(
                    torch.nn.Linear(vision_dim, vision_dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(vision_dim * 4, vision_dim),
                ),
                "ffn_norm": torch.nn.LayerNorm(vision_dim),
            }))

        self.final_norm = torch.nn.LayerNorm(vision_dim)
        self.output_proj = torch.nn.Linear(vision_dim, llm_dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # (batch, num_patches, vision_dim) → (batch, num_latents, llm_dim)
        batch_size = visual_tokens.shape[0]
        latents = self.latents.expand(batch_size, -1, -1).contiguous()

        for layer in self.layers:
            normed_media   = layer["norm_media"](visual_tokens)
            normed_latents = layer["norm_latents"](latents)

            # K/V from concat of media + latents (the Flamingo trick)
            kv_input = torch.cat((normed_media, normed_latents), dim=1)
            attn_out, _ = layer["attn"](normed_latents, kv_input, kv_input)
            latents = latents + attn_out

            latents = latents + layer["ffn"](layer["ffn_norm"](latents))

        latents = self.final_norm(latents)
        return self.output_proj(latents)


PROJECTION_REGISTRY = {
    "mlp":       MLPProjection,
    "qformer":   QFormerProjection,
    "perceiver": PerceiverResamplerProjection,
}


class MultiModalQwen(torch.nn.Module):

    def __init__(self, projection_type: str = "mlp"):
        super().__init__()
        if projection_type not in PROJECTION_REGISTRY:
            raise ValueError(
                f"Unknown projection: '{projection_type}'. "
                f"Options: {list(PROJECTION_REGISTRY.keys())}"
            )

        self.vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.image_processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        vision_dim = self.vision_model.config.hidden_size
        llm_dim = self.llm.config.hidden_size
        self.projection = PROJECTION_REGISTRY[projection_type](vision_dim, llm_dim)

        logger.info("Vision dim: {} | LLM dim: {} | Projection: {}",
                    vision_dim, llm_dim, projection_type)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        visual_tokens = self.vision_model(pixel_values=pixel_values).last_hidden_state
        projected = self.projection(visual_tokens)
        num_visual = projected.shape[1]

        # Embed text, prepend visual tokens
        text_embeds = self.llm.model.embed_tokens(input_ids)
        combined_embeds = torch.cat([projected, text_embeds], dim=1)

        # Extend attention mask: visual positions are all attended to
        visual_mask = torch.ones(
            projected.shape[0], num_visual,
            dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # Extend labels: visual positions masked so loss is text-only
        visual_labels = torch.full(
            (labels.shape[0], num_visual), -100,
            dtype=labels.dtype, device=labels.device
        )
        combined_labels = torch.cat([visual_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )


if __name__ == "__main__":
    # Example usage
    logger.info("Initializing MultiModalQwen model...")
    model = MultiModalQwen()

    # Create dataloader
    logger.info("Creating training dataloader...")
    train_loader = create_dataloader(
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        split="train",
        batch_size=4,
        shuffle=True,
        num_workers=2,
        max_length=512
    )

    # Test the dataloader
    logger.info("Testing dataloader...")
    batch = next(iter(train_loader))

    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Pixel values shape: {batch['pixel_values'].shape}")
    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
    logger.info(f"Labels shape: {batch['labels'].shape}")
    logger.info(f"Sample caption: {batch['caption'][0]}")

    # Decode a sample to see the formatted conversation
    sample_text = model.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
    logger.info(f"Formatted conversation:\n{sample_text}")