"""Projection layers for vision-language models."""
import torch
import torch.nn as nn


class MLPProjection(nn.Module):
    """Two-layer MLP projection (LLaVA-1.5 style).
    Linear(vision_dim → llm_dim) → GELU → Linear(llm_dim → llm_dim).
    Preserves all patch tokens. Both linear layers use llm_dim — there is
    no expanded intermediate dimension.

    Paper: https://arxiv.org/abs/2310.03744 (Liu et al., 2023 - Improved Baselines with Visual Instruction Tuning)
    """
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # (batch, num_patches, vision_dim) → (batch, num_patches, llm_dim)
        return self.net(visual_tokens)


class QFormerProjection(nn.Module):
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
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, vision_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(
                    vision_dim, num_heads, batch_first=True),
                "self_norm": nn.LayerNorm(vision_dim),
                "ffn": nn.Sequential(
                    nn.Linear(vision_dim, vision_dim * 4),
                    nn.GELU(),
                    nn.Linear(vision_dim * 4, vision_dim),
                ),
                "ffn_norm": nn.LayerNorm(vision_dim),
            })
            # Cross-attention only on designated layers
            if i % cross_attn_every == 0:
                layer["cross_attn"] = nn.MultiheadAttention(
                    vision_dim, num_heads, batch_first=True)
                layer["cross_norm"] = nn.LayerNorm(vision_dim)
            self.layers.append(layer)

        self.output_proj = nn.Linear(vision_dim, llm_dim)

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


class PerceiverResamplerProjection(nn.Module):
    """Perceiver Resampler projection (Flamingo style).
    K and V are derived from cat(visual_tokens, latents), so latents implicitly
    attend to each other within the same cross-attention op — no separate
    self-attention block. Pre-norm throughout, matching the Flamingo source.

    Paper: https://arxiv.org/abs/2204.14198 (Alayrac et al., 2022 - Flamingo: a Visual Language Model for Few-Shot Learning)
    """
    def __init__(self, vision_dim: int, llm_dim: int,
                 num_latents: int = 64, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, vision_dim))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "norm_media":   nn.LayerNorm(vision_dim),
                "norm_latents": nn.LayerNorm(vision_dim),
                "attn":         nn.MultiheadAttention(
                    vision_dim, num_heads, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(vision_dim, vision_dim * 4),
                    nn.GELU(),
                    nn.Linear(vision_dim * 4, vision_dim),
                ),
                "ffn_norm": nn.LayerNorm(vision_dim),
            }))

        self.final_norm = nn.LayerNorm(vision_dim)
        self.output_proj = nn.Linear(vision_dim, llm_dim)

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


# Registry for easy access
PROJECTION_REGISTRY = {
    "mlp": MLPProjection,
    "qformer": QFormerProjection,
    "perceiver": PerceiverResamplerProjection,
}


def create_projection(projection_type: str, vision_dim: int, llm_dim: int, **kwargs) -> nn.Module:
    """Factory function to create a projection layer.
    
    Args:
        projection_type: One of 'mlp', 'qformer', 'perceiver'
        vision_dim: Dimension of vision encoder output
        llm_dim: Dimension of LLM input
        **kwargs: Additional arguments passed to projection constructor
    
    Returns:
        Initialized projection layer
    """
    if projection_type not in PROJECTION_REGISTRY:
        raise ValueError(
            f"Unknown projection type: '{projection_type}'. "
            f"Available: {list(PROJECTION_REGISTRY.keys())}"
        )
    
    return PROJECTION_REGISTRY[projection_type](vision_dim, llm_dim, **kwargs)
