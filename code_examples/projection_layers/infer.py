"""Inference script for multimodal model."""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import torch
import matplotlib.pyplot as plt

from config import Config
from multimodal_model import MultiModalModel
from dataset import CocoCaptionSamples, ImagePrefetchCache


def _collect_images(image_dir: Union[str, Path]) -> List[Path]:
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    return sorted(image_paths)


def _save_results(results: List[dict], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def _save_caption_plot(image, dataset_caption: Optional[str], pred_caption: str,
                       output_path: Path) -> None:
    # Use a larger figure and better layout
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")
    
    # Clean captions - replace problematic Unicode characters
    def clean_text(text):
        if not text:
            return text
        # Replace fullwidth parentheses and other common Unicode issues
        replacements = {
            '\uff08': '(',  # Fullwidth left parenthesis
            '\uff09': ')',  # Fullwidth right parenthesis
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    # Build caption text with clear visual separation
    caption_lines = []
    
    if dataset_caption:
        gt_clean = clean_text(dataset_caption)
        # Wrap long GT captions
        if len(gt_clean) > 80:
            gt_clean = gt_clean[:77] + "..."
        caption_lines.append(f"GT: {gt_clean}")
        caption_lines.append("")  # Empty line for spacing
    
    pred_clean = clean_text(pred_caption)
    # Wrap long prediction captions
    if len(pred_clean) > 80:
        pred_clean = pred_clean[:77] + "..."
    caption_lines.append(f"Pred: {pred_clean}")
    
    # Create the caption with better formatting
    full_caption = "\n".join(caption_lines)
    
    # Use text with background for better readability
    ax.text(
        0.5, -0.02, full_caption,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
        family='sans-serif',
        wrap=True
    )
    
    # Adjust layout to make room for caption
    plt.subplots_adjust(bottom=0.15)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--projection_type",
        type=str,
        default="mlp",
        choices=["mlp", "qformer", "perceiver"],
        help="Type of projection layer to use",
    )
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    parser.add_argument("--plot_dir", type=str, default="outputs/infer_plots",
                        help="Directory to save caption plots")
    parser.add_argument("--use_coco", action="store_true",
                        help="Use COCO dataset captions for comparison")
    parser.add_argument("--coco_split", type=str, default="validation",
                        choices=["train", "validation"],
                        help="COCO split for captions")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of COCO samples to run when using --use_coco")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--no_sample", action="store_true", help="Disable sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    if args.use_coco:
        if args.image or args.image_dir:
            parser.error("--use_coco cannot be combined with --image/--image_dir")
    else:
        if not args.image and not args.image_dir:
            parser.error("Either --image or --image_dir must be provided")
        if args.image and args.image_dir:
            parser.error("Cannot provide both --image and --image_dir")

    config = Config()
    config.device = args.device if torch.cuda.is_available() else "cpu"
    config.max_new_tokens = args.max_new_tokens
    config.temperature = args.temperature
    config.top_p = args.top_p
    if args.do_sample and args.no_sample:
        parser.error("Cannot set both --do_sample and --no_sample")
    config.do_sample = not args.no_sample if (args.do_sample or args.no_sample) else config.do_sample

    print(f"Using device: {config.device}")
    print(f"Checkpoint: {args.checkpoint or 'none (base model)'}")
    print(f"Projection type: {args.projection_type}")
    print(
        "Generation params: "
        f"max_new_tokens={config.max_new_tokens}, "
        f"temperature={config.temperature}, "
        f"top_p={config.top_p}, "
        f"do_sample={config.do_sample}"
    )

    print("\nLoading model...")
    model = MultiModalModel(config, projection_type=args.projection_type)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)
    model.eval()

    if args.checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print("\n" + "=" * 60)

    results = []
    prefetch_cache = ImagePrefetchCache(
        max_workers=config.num_workers,
        max_cached=max(2, config.prefetch_size * 2),
    )
    if args.use_coco:
        coco_samples = CocoCaptionSamples(
            split=args.coco_split,
            prefetch_size=config.prefetch_size,
            prefetch_workers=config.num_workers,
        )
        plot_dir = Path(args.plot_dir)
        for idx in range(min(args.num_samples, len(coco_samples))):
            sample = coco_samples[idx]
            caption = model.generate_caption_from_image(
                sample["image"],
                config.device,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                system_prompt=config.system_prompt,
                user_prompt=config.user_prompt,
            )
            result = {
                "image": sample["url"],
                "dataset_caption": sample["caption"],
                "caption": caption,
                "image_id": sample["image_id"],
            }
            results.append(result)
            print(f"\n[{idx + 1}/{args.num_samples}] {sample['image_id']}")
            print(f"  GT: {sample['caption']}")
            print(f"  Pred: {caption}")
            _save_caption_plot(
                sample["image"],
                sample["caption"],
                caption,
                plot_dir / f"coco_{sample['image_id']}.png",
            )
    elif args.image:
        print(f"Processing single image: {args.image}")
        print("=" * 60)
        image = prefetch_cache.get("single", str(args.image))
        caption = model.generate_caption_from_image(
            image,
            config.device,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
        )
        result = {"image": str(args.image), "caption": caption}
        results.append(result)
        print(f"\nImage: {result['image']}")
        print(f"Caption: {result['caption']}")
        _save_caption_plot(image, None, caption, Path(args.plot_dir) / "single.png")
    else:
        print(f"Processing directory: {args.image_dir}")
        print("=" * 60)
        image_paths = _collect_images(args.image_dir)
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            return
        print(f"Found {len(image_paths)} images")
        for index, image_path in enumerate(image_paths, start=1):
            print(f"\n[{index}/{len(image_paths)}] {image_path.name}")
            try:
                if index < len(image_paths):
                    prefetch_cache.prefetch(index + 1, str(image_paths[index]))
                image = prefetch_cache.get(index, str(image_path))
                caption = model.generate_caption_from_image(
                    image,
                    config.device,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    system_prompt=config.system_prompt,
                    user_prompt=config.user_prompt,
                )
                result = {"image": str(image_path), "caption": caption}
                results.append(result)
                print(f"  Caption: {caption}")
                _save_caption_plot(
                    image,
                    None,
                    caption,
                    Path(args.plot_dir) / f"{image_path.stem}.png",
                )
            except Exception as exc:
                print(f"  Error: {exc}")
                results.append({"image": str(image_path), "caption": "", "error": str(exc)})

    if args.output:
        _save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
