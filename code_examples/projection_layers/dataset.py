"""Datasets and image prefetch utilities."""
from collections import OrderedDict
import concurrent.futures
from io import BytesIO
from typing import Callable, Optional

from datasets import load_dataset
from PIL import Image
import requests
from torch.utils.data import Dataset


def _load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(source).convert("RGB")


class ImagePrefetchCache:
    """Async prefetch cache for images (URL or local path)."""

    def __init__(self, max_workers: int = 4, max_cached: int = 64,
                 load_fn: Optional[Callable[[str], Image.Image]] = None):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures = OrderedDict()
        self._max_cached = max_cached
        self._load_fn = load_fn or _load_image

    def prefetch(self, key, source: str) -> None:
        if key in self._futures:
            return
        while len(self._futures) >= self._max_cached:
            _, evicted = self._futures.popitem(last=False)
            evicted.cancel()
        self._futures[key] = self._executor.submit(self._load_fn, source)

    def get(self, key, source: str, timeout: int = 30) -> Image.Image:
        self.prefetch(key, source)
        future = self._futures.pop(key)
        return future.result(timeout=timeout)


class CocoCaptionsDataset(Dataset):
    """COCO Captions dataset formatted for chat-style captioning."""

    def __init__(self, tokenizer, image_processor, split="train", max_length=512,
                 system_prompt="You are a helpful assistant.",
                 user_prompt="Describe this image.",
                 prefetch_size: int = 16, prefetch_workers: int = 4,
                 max_samples: int = -1):
        self.dataset = load_dataset("yerevann/coco-karpathy", split=split)
        if max_samples > 0:
            actual_samples = min(max_samples, len(self.dataset))
            self.dataset = self.dataset.select(range(actual_samples))
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.prefetch_size = prefetch_size
        self._prefetch_cache = ImagePrefetchCache(
            max_workers=prefetch_workers,
            max_cached=prefetch_size * 2,
        )
        self._max_retries = 5

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, idx):
        for offset in range(1, self.prefetch_size + 1):
            next_idx = (idx + offset) % len(self)
            self._prefetch_cache.prefetch(next_idx, self.dataset[next_idx]["url"])

        current_idx = idx
        for _ in range(self._max_retries):
            try:
                url = self.dataset[current_idx]["url"]
                image = self._prefetch_cache.get(current_idx, url)
                return image, current_idx
            except Exception:
                current_idx = (current_idx + 1) % len(self)

        raise RuntimeError(f"Failed to load image after {self._max_retries} retries")

    def __getitem__(self, idx):
        image, item_idx = self._load_image(idx)
        item = self.dataset[item_idx]

        image_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        captions = item["sentences"]
        caption = captions[item_idx % len(captions)]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
            {"role": "assistant", "content": caption},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        prompt_text = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        prompt_length = prompt_encoding["input_ids"].size(1)
        labels[:prompt_length] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption,
        }


class CocoCaptionSamples(Dataset):
    """COCO samples for inference (PIL image + caption)."""

    def __init__(self, split="validation", prefetch_size: int = 16, prefetch_workers: int = 4):
        self.dataset = load_dataset("yerevann/coco-karpathy", split=split)
        self.prefetch_size = prefetch_size
        self._prefetch_cache = ImagePrefetchCache(
            max_workers=prefetch_workers,
            max_cached=prefetch_size * 2,
        )
        self._max_retries = 5

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, idx):
        for offset in range(1, self.prefetch_size + 1):
            next_idx = (idx + offset) % len(self)
            self._prefetch_cache.prefetch(next_idx, self.dataset[next_idx]["url"])

        current_idx = idx
        for _ in range(self._max_retries):
            try:
                url = self.dataset[current_idx]["url"]
                image = self._prefetch_cache.get(current_idx, url)
                return image, current_idx
            except Exception:
                current_idx = (current_idx + 1) % len(self)

        raise RuntimeError(f"Failed to load image after {self._max_retries} retries")

    def __getitem__(self, idx):
        image, item_idx = self._load_image(idx)
        item = self.dataset[item_idx]
        captions = item["sentences"]
        caption = captions[item_idx % len(captions)]

        return {
            "image": image,
            "caption": caption,
            "url": item["url"],
            "image_id": item.get("imgid", item_idx),
        }
