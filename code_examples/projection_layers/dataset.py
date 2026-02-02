"""Datasets and image prefetch utilities."""
import asyncio
from collections import OrderedDict
from io import BytesIO
import os
import threading
from typing import Optional

# Disable tkinter backend for PIL to prevent multiprocessing issues
os.environ.setdefault('MPLBACKEND', 'Agg')

import aiohttp
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

# Explicitly prevent PIL from using tkinter
Image.USE_CFFI_ACCESS = False


async def _load_image_async(session: aiohttp.ClientSession, source: str) -> Image.Image:
    """Load image from URL or local path asynchronously."""
    if source.startswith("http://") or source.startswith("https://"):
        async with session.get(source, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            content = await resp.read()
            img = Image.open(BytesIO(content)).convert("RGB")
            # Force load image data to break any tkinter references
            img.load()
            return img
    # For local paths, run in thread to avoid blocking
    loop = asyncio.get_event_loop()
    img = await loop.run_in_executor(None, lambda: Image.open(source).convert("RGB"))
    img.load()  # Force load to break any lazy-loading references
    return img


class ImagePrefetchCache:
    """Async prefetch cache for images using asyncio and aiohttp for high-performance concurrent downloads."""

    def __init__(self, max_workers: int = 16, max_cached: int = 256):
        self._max_workers = max_workers
        self._max_cached = max_cached
        self._tasks = OrderedDict()
        self._semaphore = None
        self._session = None
        self._loop = None
        self._thread = None
        self._started = False
        self._lock = threading.Lock()

    def _start_event_loop(self):
        """Start background event loop in a dedicated thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_started(self):
        """Lazily initialize the async event loop and aiohttp session."""
        if self._started:
            return

        with self._lock:
            if self._started:
                return

            # Create event loop in background thread
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._start_event_loop, daemon=True)
            self._thread.start()

            # Create session and semaphore in the event loop
            future = asyncio.run_coroutine_threadsafe(self._init_async(), self._loop)
            future.result()
            self._started = True

    async def _init_async(self):
        """Initialize async components."""
        self._semaphore = asyncio.Semaphore(self._max_workers)
        # Connection pool with keep-alive and limits
        connector = aiohttp.TCPConnector(
            limit=self._max_workers * 2,
            limit_per_host=10,
            ttl_dns_cache=300,
        )
        self._session = aiohttp.ClientSession(connector=connector)

    async def _fetch_image(self, source: str) -> Image.Image:
        """Fetch single image with semaphore limit."""
        async with self._semaphore:
            return await _load_image_async(self._session, source)

    def prefetch(self, key, source: str) -> None:
        """Schedule async image download."""
        self._ensure_started()

        if key in self._tasks:
            return

        # Evict oldest if cache full
        while len(self._tasks) >= self._max_cached:
            _, evicted = self._tasks.popitem(last=False)
            evicted.cancel()

        # Schedule download task
        coro = self._fetch_image(source)
        task = asyncio.run_coroutine_threadsafe(coro, self._loop)
        self._tasks[key] = task

    def get(self, key, source: str, timeout: int = 30) -> Image.Image:
        """Get image, blocking until download completes."""
        self.prefetch(key, source)
        task = self._tasks.pop(key)
        return task.result(timeout=timeout)

    def __del__(self):
        """Cleanup resources."""
        if self._started and self._loop:
            # Schedule session cleanup
            if self._session:
                asyncio.run_coroutine_threadsafe(self._session.close(), self._loop)
            # Stop event loop
            self._loop.call_soon_threadsafe(self._loop.stop)


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
            max_cached=max(prefetch_size * 4, 256),  # At least 256 for better throughput
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
            max_cached=max(prefetch_size * 4, 256),  # At least 256 for better throughput
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
