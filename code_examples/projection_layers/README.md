# Projection Layer Experiments

Compare three multimodal projection architectures (MLP, Q-Former, Perceiver Resampler) trained on COCO image captioning.

## Setup

```bash
conda activate blog-code-examples
```

## Training

```bash
python train.py --projection_type <mlp|qformer|perceiver>
```

### All CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--projection_type` | `mlp` | Architecture: `mlp`, `qformer`, `perceiver` |
| `--freeze_vision` | `True` | Freeze vision encoder weights |
| `--freeze_llm` | `True` | Freeze LLM weights |
| `--batch_size` | `8` | Training batch size |
| `--num_epochs` | `10` | Number of training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--grad_clip` | `1.0` | Gradient clipping value |
| `--warmup_steps` | `100` | Linear warmup steps |
| `--max_length` | `512` | Max token sequence length |
| `--num_workers` | `8` | DataLoader worker processes |
| `--train_max_samples` | `1000` | Training samples (-1 for all) |
| `--val_max_samples` | `1000` | Validation samples (-1 for all) |
| `--val_interval` | `1` | Validate every N epochs |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |
| `--mixed_precision` | `True` | Enable mixed precision (AMP) |
| `--seed` | `42` | Random seed |
| `--output_dir` | `outputs` | Directory for checkpoints and logs |
| `--log_interval` | `50` | Steps between loss logging |
| `--save_interval` | `2` | Epochs between checkpoints |
| `--max_new_tokens` | `50` | Max tokens to generate during validation |
| `--temperature` | `1.0` | Sampling temperature |
| `--top_p` | `0.9` | Nucleus sampling top-p |
| `--do_sample` | `True` | Enable sampling (False = greedy) |
| `--resume` | — | Path to checkpoint to resume from |

### Examples

```bash
# Train MLP on full dataset
python train.py --projection_type mlp --train_max_samples -1 --val_max_samples -1

# Train Q-Former with custom output dir
python train.py --projection_type qformer --output_dir trained_model_qformer

# Train Perceiver on CPU
python train.py --projection_type perceiver --device cpu --mixed_precision false

# Resume training from checkpoint
python train.py --projection_type mlp --resume outputs/checkpoints/epoch_005.pt
```

Outputs saved to `--output_dir`:
- `best_model.pt` — best checkpoint by CIDEr
- `checkpoints/epoch_NNN.pt` — periodic checkpoints
- `train.log` — full training log
- `metrics.json` — per-epoch metrics history
- `training_curves.png` — loss and CIDEr plots

---

## Inference

```bash
python infer.py [options]
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--projection_type` | `mlp` | Architecture: `mlp`, `qformer`, `perceiver` |
| `--checkpoint` | — | Path to trained checkpoint (omit for untrained) |
| `--image` | — | Single image file path |
| `--image_dir` | — | Directory of images |
| `--use_coco` | — | Sample from COCO validation set |
| `--num_samples` | `8` | Number of COCO samples to run |
| `--plot_dir` | `outputs/infer_plots` | Where to save output plots |
| `--max_new_tokens` | `128` | Max tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top_p` | `0.9` | Nucleus sampling top-p |
| `--no_sample` | — | Use greedy decoding |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### Examples

```bash
# Untrained model on COCO (baseline)
python infer.py --use_coco --num_samples 6 --plot_dir outputs/untrained

# Trained MLP on COCO
python infer.py --use_coco --checkpoint trained_model_mlp/best_model.pt --projection_type mlp --num_samples 6

# Trained Q-Former on COCO
python infer.py --use_coco --checkpoint trained_model_qformer/best_model.pt --projection_type qformer --num_samples 6

# Trained Perceiver on COCO
python infer.py --use_coco --checkpoint trained_model_perceiver/best_model.pt --projection_type perceiver --num_samples 6

# Single image with trained model
python infer.py --image photo.jpg --checkpoint trained_model_mlp/best_model.pt --projection_type mlp
```

---

## Visualizations

Generates attention heatmaps from all three trained models on COCO images.

```bash
python visualize_trained_attention.py
```

Expects trained models at:
- `trained_model_mlp/best_model.pt`
- `trained_model_qformer/best_model.pt`
- `trained_model_perceiver/best_model.pt`

Outputs saved to `visualizations/`:

| File | Description |
|------|-------------|
| `trained_attention_model_comparison.png` | Side-by-side: original image, Q-Former attention overlay, Perceiver attention overlay across 3 COCO images |
| `trained_attention_qformer_specialization.png` | 8 individual Q-Former query attention maps showing how different queries specialize |
| `trained_attention_perceiver_specialization.png` | 8 individual Perceiver latent attention maps showing how different latents specialize |
| `trained_attention_grids.png` | Q-Former and Perceiver attention as raw 14x14 patch grids (no image overlay) |
