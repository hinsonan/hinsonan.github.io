#!/usr/bin/env python3
"""Generate trained attention visualizations on real COCO images.

Usage:
    conda run -n blog-code-examples python visualize_trained_attention.py
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def setup_plot_style() -> None:
    """Set up consistent plot styling."""
    plt.style.use("default")
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def visualize_trained_attention_on_coco(output_path: Path) -> None:
    """Visualize trained attention patterns on COCO images."""
    setup_plot_style()

    print("Loading trained models and COCO images...")

    from dataset import CocoCaptionSamples
    from multimodal_model import MultiModalModel
    from config import Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    coco_samples = CocoCaptionSamples(split="validation", prefetch_size=4, prefetch_workers=4)
    image_indices = [10, 25, 42]
    images = []
    captions = []

    for idx in image_indices:
        sample = coco_samples[idx]
        images.append(sample["image"])
        captions.append(sample["caption"])

    print(f"Loaded {len(images)} COCO images")

    models: Dict[str, Any] = {}
    model_dirs = {
        "mlp": Path(__file__).parent / "trained_model_mlp",
        "qformer": Path(__file__).parent / "trained_model_qformer",
        "perceiver": Path(__file__).parent / "trained_model_perceiver",
    }

    for proj_type, model_dir in model_dirs.items():
        checkpoint_path = model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found, skipping {proj_type}")
            continue

        print(f"Loading {proj_type} model from {checkpoint_path}...")
        config = Config()
        config.projection_type = proj_type
        model = MultiModalModel(config, projection_type=proj_type).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        models[proj_type] = model
        print(f"  ✓ Loaded {proj_type} model (CIDEr: {checkpoint['metrics'].get('cider', 0):.4f})")

    if not models:
        print("No trained models found! Skipping trained attention visualization.")
        return

    print("\nGenerating visualizations...")

    if len(models) >= 2:
        print("  Creating model comparison visualization...")
        fig = plt.figure(figsize=(18, 6 * len(images)))

        for img_idx, (image, caption) in enumerate(zip(images, captions)):
            row_start = img_idx * 3

            ax = plt.subplot(len(images), 3, row_start + 1)
            ax.imshow(image)
            ax.set_title("MLP: No Attention\n(Direct mapping)", fontsize=11, fontweight="bold")
            ax.axis("off")

            if "mlp" in models:
                ax.text(0.5, -0.05, "Model compresses: 196 → 196 tokens", transform=ax.transAxes, ha="center", fontsize=9, style="italic")

            pixel_values = None
            visual_tokens = None
            qformer_ax = None

            for proj_type in ["qformer", "perceiver"]:
                if proj_type not in models:
                    continue

                model = models[proj_type]

                if pixel_values is None:
                    pixel_values = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
                    with torch.no_grad():
                        vision_outputs = model.vision_model(pixel_values=pixel_values)
                        visual_tokens = vision_outputs.last_hidden_state

                visual_tokens = visual_tokens if visual_tokens is not None else torch.empty(1, 196, 768, device=device)

                col_idx = 2 if proj_type == "qformer" else 3
                ax = plt.subplot(len(images), 3, row_start + col_idx)
                if col_idx == 2:
                    qformer_ax = ax

                with torch.no_grad():
                    if proj_type == "qformer":
                        qformer_proj = model.projection
                        batch_size = visual_tokens.shape[0]
                        queries = qformer_proj.query_tokens.expand(batch_size, -1, -1)
                        attn_map = np.zeros((14, 14))

                        for layer in qformer_proj.layers:
                            sa_out, _ = layer["self_attn"](queries, queries, queries)
                            queries = layer["self_norm"](queries + sa_out)

                            if "cross_attn" in layer:
                                _, ca_weights = layer["cross_attn"](queries, visual_tokens, visual_tokens)

                                if img_idx == 0:
                                    attn_sums = ca_weights[0].sum(dim=-1).cpu().numpy()
                                    print("      Q-Former model comparison attention verification:")
                                    print(f"        - Attention sums: min={attn_sums.min():.6f}, max={attn_sums.max():.6f}")

                                avg_attn = ca_weights[0].mean(dim=0).cpu().numpy()
                                attn_map = avg_attn.reshape(14, 14)
                                break

                        ax.imshow(image, alpha=0.6)
                        attn_overlay = ax.imshow(
                            attn_map,
                            cmap="hot",
                            alpha=0.5,
                            extent=(0, image.size[0], image.size[1], 0),
                            interpolation="bilinear",
                        )
                        ax.set_title("Q-Former Attention\n(Avg across 32 queries)", fontsize=11, fontweight="bold")
                        ax.axis("off")

                        cbar = plt.colorbar(attn_overlay, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label("Attention", fontsize=9)

                        ax.text(0.5, -0.05, "Trained model | Compression: 196 → 32", transform=ax.transAxes, ha="center", fontsize=9, style="italic")

                    elif proj_type == "perceiver":
                        perceiver_proj = model.projection
                        batch_size = visual_tokens.shape[0]
                        latents = perceiver_proj.latents.expand(batch_size, -1, -1).contiguous()

                        layer = perceiver_proj.layers[0]
                        normed_media = layer["norm_media"](visual_tokens)
                        normed_latents = layer["norm_latents"](latents)
                        kv_input = torch.cat((normed_media, normed_latents), dim=1)
                        _, attn_weights = layer["attn"](normed_latents, kv_input, kv_input)

                        if img_idx == 0:
                            attn_sums = attn_weights[0].sum(dim=-1).cpu().numpy()
                            print("      Perceiver model comparison attention verification:")
                            print(f"        - Attention sums: min={attn_sums.min():.6f}, max={attn_sums.max():.6f}")

                        visual_attn = attn_weights[0, :, :196].mean(dim=0).cpu().numpy()
                        attn_map = visual_attn.reshape(14, 14)

                        ax.imshow(image, alpha=0.6)
                        attn_overlay = ax.imshow(
                            attn_map,
                            cmap="viridis",
                            alpha=0.5,
                            extent=(0, image.size[0], image.size[1], 0),
                            interpolation="bilinear",
                        )
                        ax.set_title("Perceiver Attention\n(Avg across 64 latents)", fontsize=11, fontweight="bold")
                        ax.axis("off")

                        cbar = plt.colorbar(attn_overlay, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label("Attention", fontsize=9)

                        ax.text(0.5, -0.05, "Trained model | Compression: 196 → 64", transform=ax.transAxes, ha="center", fontsize=9, style="italic")

            if img_idx == 0:
                fig.text(0.5, 0.97, "Model Comparison: Same COCO Image Through Different Projection Layers", ha="center", fontsize=14, fontweight="bold")

            # Add caption to the middle subplot of the row
            if qformer_ax is not None:
                ax_mid = qformer_ax
            else:
                # Fallback if qformer wasn't plotted
                ax_mid = plt.subplot(len(images), 3, row_start + 2)
                ax_mid.axis("off")

            ax_mid.text(
                0.5,
                -0.12,
                f"Caption: \"{caption}\"",
                transform=ax_mid.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                style="italic",
                wrap=True,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
                zorder=10
            )

        plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
        plt.savefig(output_path / "trained_attention_model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Model comparison visualization")

    if "qformer" in models:
        print("  Creating Q-Former query specialization visualization...")
        model = models["qformer"]
        image = images[0]

        pixel_values = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        ca_weights = torch.zeros(1, 32, 196, device=device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            visual_tokens = vision_outputs.last_hidden_state

            qformer_proj = model.projection
            batch_size = visual_tokens.shape[0]
            queries = qformer_proj.query_tokens.expand(batch_size, -1, -1)

            for layer in qformer_proj.layers:
                sa_out, _ = layer["self_attn"](queries, queries, queries)
                queries = layer["self_norm"](queries + sa_out)

                if "cross_attn" in layer:
                    _, ca_weights = layer["cross_attn"](queries, visual_tokens, visual_tokens)

                    attn_sums = ca_weights[0].sum(dim=-1).cpu().numpy()
                    print("      Q-Former attention verification:")
                    print(f"        - Shape: {ca_weights.shape} (batch, queries, patches)")
                    print(f"        - Attention sums: min={attn_sums.min():.6f}, max={attn_sums.max():.6f}, mean={attn_sums.mean():.6f}")
                    print(f"        - All close to 1.0? {np.allclose(attn_sums, 1.0, atol=1e-5)}")
                    break

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        query_indices = [0, 4, 8, 12, 16, 20, 24, 28]

        for idx, query_idx in enumerate(query_indices):
            ax = axes[idx // 4, idx % 4]

            attn = ca_weights[0, query_idx].cpu().numpy().reshape(14, 14)

            ax.imshow(image, alpha=0.5)
            im = ax.imshow(
                attn,
                cmap="hot",
                alpha=0.6,
                extent=(0, image.size[0], image.size[1], 0),
                interpolation="bilinear",
            )
            ax.set_title(f"Query #{query_idx}", fontsize=11, fontweight="bold")
            ax.axis("off")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Q-Former Query Specialization: Different Queries Focus on Different Regions", fontsize=14, fontweight="bold", y=0.98)
        fig.text(0.5, 0.02, f"Caption: \"{captions[0]}\"", ha="center", fontsize=11, style="italic", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

        plt.tight_layout(rect=(0, 0.04, 1, 0.96))
        plt.savefig(output_path / "trained_attention_qformer_specialization.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Q-Former query specialization")

    if "perceiver" in models:
        print("  Creating Perceiver latent specialization visualization...")
        model = models["perceiver"]
        image = images[0]

        pixel_values = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            visual_tokens = vision_outputs.last_hidden_state

            perceiver_proj = model.projection
            batch_size = visual_tokens.shape[0]
            latents = perceiver_proj.latents.expand(batch_size, -1, -1).contiguous()

            layer = perceiver_proj.layers[0]
            normed_media = layer["norm_media"](visual_tokens)
            normed_latents = layer["norm_latents"](latents)
            kv_input = torch.cat((normed_media, normed_latents), dim=1)
            _, attn_weights = layer["attn"](normed_latents, kv_input, kv_input)

            attn_sums = attn_weights[0].sum(dim=-1).cpu().numpy()
            print("      Perceiver attention verification (specialization viz):")
            print(f"        - Shape: {attn_weights.shape} (batch, latents, total_tokens)")
            print(f"        - Attention sums: min={attn_sums.min():.6f}, max={attn_sums.max():.6f}, mean={attn_sums.mean():.6f}")
            print(f"        - All close to 1.0? {np.allclose(attn_sums, 1.0, atol=1e-5)}")

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        latent_indices = [0, 8, 16, 24, 32, 40, 48, 56]

        for idx, latent_idx in enumerate(latent_indices):
            ax = axes[idx // 4, idx % 4]

            attn = attn_weights[0, latent_idx, :196].cpu().numpy().reshape(14, 14)

            ax.imshow(image, alpha=0.5)
            im = ax.imshow(
                attn,
                cmap="viridis",
                alpha=0.6,
                extent=(0, image.size[0], image.size[1], 0),
                interpolation="bilinear",
            )
            ax.set_title(f"Latent #{latent_idx}", fontsize=11, fontweight="bold")
            ax.axis("off")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Perceiver Latent Specialization: Different Latents Focus on Different Regions", fontsize=14, fontweight="bold", y=0.98)
        fig.text(0.5, 0.02, f"Caption: \"{captions[0]}\"", ha="center", fontsize=11, style="italic", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

        plt.tight_layout(rect=(0, 0.04, 1, 0.96))
        plt.savefig(output_path / "trained_attention_perceiver_specialization.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Perceiver latent specialization")

    if "qformer" in models and "perceiver" in models:
        print("  Creating attention grid visualization...")
        fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))

        if len(images) == 1:
            axes = axes.reshape(1, -1)

        for img_idx, (image, caption) in enumerate(zip(images, captions)):
            ax = axes[img_idx, 0]
            ax.imshow(image)
            ax.set_title("Original COCO Image", fontsize=11, fontweight="bold")
            ax.axis("off")

            model = models["qformer"]
            qformer_attn = np.zeros((14, 14))
            pixel_values = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                vision_outputs = model.vision_model(pixel_values=pixel_values)
                visual_tokens = vision_outputs.last_hidden_state

                qformer_proj = model.projection
                batch_size = visual_tokens.shape[0]
                queries = qformer_proj.query_tokens.expand(batch_size, -1, -1)

                for layer in qformer_proj.layers:
                    sa_out, _ = layer["self_attn"](queries, queries, queries)
                    queries = layer["self_norm"](queries + sa_out)

                    if "cross_attn" in layer:
                        _, ca_weights = layer["cross_attn"](queries, visual_tokens, visual_tokens)

                        if img_idx == 0:
                            attn_sums = ca_weights[0].sum(dim=-1).cpu().numpy()
                            print("      Q-Former grid attention verification:")
                            print(f"        - Attention sums: min={attn_sums.min():.6f}, max={attn_sums.max():.6f}")

                        qformer_attn = ca_weights[0].mean(dim=0).cpu().numpy().reshape(14, 14)
                        break

            ax = axes[img_idx, 1]
            im = ax.imshow(qformer_attn, cmap="hot", aspect="equal")
            ax.set_title("Q-Former Attention\n(14×14 grid)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Patch Column", fontsize=9)
            ax.set_ylabel("Patch Row", fontsize=9)
            ax.grid(which="both", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            model = models["perceiver"]
            pixel_values = model.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                vision_outputs = model.vision_model(pixel_values=pixel_values)
                visual_tokens = vision_outputs.last_hidden_state

                perceiver_proj = model.projection
                batch_size = visual_tokens.shape[0]
                latents = perceiver_proj.latents.expand(batch_size, -1, -1).contiguous()

                layer = perceiver_proj.layers[0]
                normed_media = layer["norm_media"](visual_tokens)
                normed_latents = layer["norm_latents"](latents)
                kv_input = torch.cat((normed_media, normed_latents), dim=1)
                _, attn_weights = layer["attn"](normed_latents, kv_input, kv_input)

                if img_idx == 0:
                    attn_sums = attn_weights[0].sum(dim=-1).cpu().numpy()
                    print("      Perceiver grid attention verification:")
                    print(f"        - Attention sums: min={attn_sums.min():.6f}, max={attn_sums.max():.6f}")

                perceiver_attn = attn_weights[0, :, :196].mean(dim=0).cpu().numpy().reshape(14, 14)

            ax = axes[img_idx, 2]
            im = ax.imshow(perceiver_attn, cmap="viridis", aspect="equal")
            ax.set_title("Perceiver Attention\n(14×14 grid)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Patch Column", fontsize=9)
            ax.set_ylabel("Patch Row", fontsize=9)
            ax.grid(which="both", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add caption to the middle subplot
            ax_mid = axes[img_idx, 1]
            ax_mid.text(
                0.5,
                -0.25,
                f"Caption: \"{caption}\"",
                transform=ax_mid.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                style="italic",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
                zorder=10
            )

        fig.suptitle("Attention Patterns as 14×14 Grids on COCO Images", fontsize=14, fontweight="bold", y=0.995)

        plt.tight_layout(rect=(0, 0, 1, 0.99), h_pad=3.0)
        plt.savefig(output_path / "trained_attention_grids.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("    ✓ Attention grid visualization")

    print("\n✓ All trained attention visualizations complete!")


def main() -> None:
    """Generate trained attention visualizations."""
    output_path = Path(__file__).parent / "visualizations"
    output_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("Generating Trained Attention Visualizations on COCO Images")
    print("=" * 70)
    print()

    visualize_trained_attention_on_coco(output_path)

    print()
    print("=" * 70)
    print(f"✓ All visualizations saved to: {output_path}")
    print("=" * 70)
    print()
    print("Generated files:")
    for img in sorted(output_path.glob("trained_attention_*.png")):
        print(f"  - {img.name}")
    print()


if __name__ == "__main__":
    main()
