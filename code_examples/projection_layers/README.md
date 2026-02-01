# Image Captioning Inference

## Quick Start

```bash
# Untrained model (bad captions)
python infer.py --use_coco --num_samples 6 --plot_dir outputs/before

# After training (good captions)  
python infer.py --use_coco --checkpoint outputs/best_model.pt --num_samples 6 --plot_dir outputs/after
```

## CLI Usage

```bash
python infer.py [options]
```

### Key Arguments

- `--image PATH` - Single image
- `--image_dir PATH` - Directory of images
- `--use_coco` - Use COCO dataset with GT captions
- `--checkpoint PATH` - Model checkpoint (optional)
- `--projection_type {mlp,qformer,perceiver}` - Architecture (default: mlp)
- `--plot_dir PATH` - Where to save plots (default: outputs/infer_plots)
- `--num_samples N` - Samples when using --use_coco (default: 8)
- `--max_new_tokens N` - Max tokens (default: 128)
- `--temperature FLOAT` - Sampling temp (default: 1.0)
- `--top_p FLOAT` - Top-p sampling (default: 0.9)
- `--no_sample` - Greedy decoding
- `--device {cuda,cpu}` - Device to use

### Examples

**Single image:**
```bash
python infer.py --image photo.jpg --checkpoint best_model.pt
```

**Batch:**
```bash
python infer.py --image_dir photos/ --checkpoint best_model.pt --output results.json
```

**Different projection:**
```bash
python infer.py --use_coco --checkpoint qformer.pt --projection_type qformer
```

## Output

Plots saved to `--plot_dir` showing:
- Image
- GT caption (COCO only)
- Generated caption

Results optionally saved as JSON with `--output`.

## Training

```bash
# Train with MLP
python train.py --projection_type mlp

# Train with Q-Former
python train.py --projection_type qformer

# Resume training
python train.py --resume outputs/checkpoints/epoch_005.pt --projection_type mlp
```

Outputs: `outputs/best_model.pt`, `outputs/checkpoints/`, logs, metrics, plots.
