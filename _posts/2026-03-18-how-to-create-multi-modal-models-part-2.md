---
layout: post
title: "How to Create Multi Modal Models Part 2"
date: 2026-02-07
categories: ML
---

It's time to take the lessons from the last blog post and make ourselves a genuine multi modal model. This is all the rage these days. We are going to train 3 different projection layers and see how they allow the model to understand images. The reason we are going to do these different projection layers is this is an experiment that you can run locally if you have a decent gpu. Here are the three methods we will use.

* Multi-Layer Projection
* Q Former
* Perceiver Resampler

## Multi-Layer Projection

This is straight out of the [LLaVA](https://llava-vl.github.io/) playbook. These researchers and the community really put multi modal models on the map. Many research institutes and companies took their MLP projection technique and continued to iterate and refine it.

<div class="mermaid">
flowchart TB
    subgraph Input["Input"]
        VT["Visual Tokens<br/>(batch, num_patches, vision_dim)"]
    end

    subgraph MLP["Two-Layer MLP Projection  ·  LLaVA-1.5 Style"]
        L1["Linear Layer 1<br/>vision_dim → llm_dim"]
        GELU["GELU Activation"]
        L2["Linear Layer 2<br/>llm_dim → llm_dim"]
    end

    subgraph Output["Output"]
        PT["Projected Tokens<br/>(batch, num_patches, llm_dim)"]
    end

    VT --> L1
    L1 --> GELU
    GELU --> L2
    L2 --> PT

    style Input fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style MLP fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style Output fill:#1e4d4d,stroke:#50c878,stroke-width:3px,color:#fff
    style VT fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style L1 fill:#7b341e,stroke:#fc8181,stroke-width:2px,color:#fff
    style GELU fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style L2 fill:#7b341e,stroke:#fc8181,stroke-width:2px,color:#fff
    style PT fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
</div>

This approach has a few advantages.

1) Very simple and fast to train

2) All of the patches are preserved so no information is thrown away

3) Simple to debug and conceptualize

The downside of this method is that it blows the context up since we keep all the vision tokens. This can help when the task needs all those details but many times you do not need all this information

## Q Former

This method was used in [BLIP-2](https://arxiv.org/pdf/2301.12597) and essentially acts as learned queries/questions to derive from the image.

<div class="mermaid">
flowchart TB
    subgraph Input["Input"]
        VT["Visual Tokens<br/>(batch, num_patches, vision_dim)"]
        QT["Learnable Query Tokens<br/>(1, num_queries=32, vision_dim)"]
    end

    subgraph Layer["Transformer Layer  ×12"]
        direction TB
        SA["Self-Attention<br/>queries attend to queries"]
        SN["LayerNorm"]
        CA["Cross-Attention  ·  every other layer<br/>queries attend to visual tokens"]
        CN["LayerNorm"]
        FFN["FFN  ·  Linear → GELU → Linear<br/>vision_dim → vision_dim×4 → vision_dim"]
        FN["LayerNorm"]

        SA --> SN --> CA --> CN --> FFN --> FN
    end

    subgraph Output["Output"]
        OP["Linear Projection<br/>vision_dim → llm_dim"]
        PT["Projected Tokens<br/>(batch, num_queries=32, llm_dim)"]
        OP --> PT
    end

    VT -->|"keys & values"| CA
    QT --> SA
    FN -->|"next layer"| SA
    FN --> OP

    style Input fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style Layer fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style Output fill:#1e4d4d,stroke:#50c878,stroke-width:3px,color:#fff
    style VT fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style QT fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style SA fill:#7b341e,stroke:#fc8181,stroke-width:2px,color:#fff
    style SN fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style CA fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style CN fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style FFN fill:#7b341e,stroke:#fc8181,stroke-width:2px,color:#fff
    style FN fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style OP fill:#276749,stroke:#68d391,stroke-width:2px,color:#fff
    style PT fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
</div>

The benefits of this method are:

1) compression of information based on a fixed number of queries

2) You can scale up vision encoder and still have a fast inference with a fixed number of queries

3) You can refine these patches through self & cross attention before feeding it to the llm

This method has its cons. It is harder and longer to train since you have to train many more parameters and mature the query space. No matter what you do you are losing information by condensing down to a fixed number of queries.

## Perceiver Resampler

Made famous by our favorite bird [Flamingo](https://arxiv.org/pdf/2204.14198) this method collapses the self attention and cross attention operations into one operation. This allows for the visual tokens and latents to be concatenated together and each latent can attend to the image patches and other latents.

<div class="mermaid">
flowchart TB
    subgraph Input["Input"]
        VT["Visual Tokens\n(batch, num_patches, vision_dim)"]
        LA["Learnable Latents\n(1, num_latents=64, vision_dim)"]
    end

    subgraph Layer["Perceiver Layer  ×2"]
        direction TB
        NM["LayerNorm  ·  visual tokens"]
        NL["LayerNorm  ·  latents"]
        CAT["Concat\nnormed visual tokens + normed latents\n→ keys & values"]
        CA["Cross-Attention\nQ = normed latents\nK,V = concat above"]
        RES["Residual Add\nlatents = latents + attn_out"]
        FFN["FFN  ·  Linear → GELU → Linear\nvision_dim → vision_dim×4 → vision_dim"]
        FRES["Residual Add + LayerNorm"]

        NM --> CAT
        NL --> CA
        NL --> CAT
        CAT --> CA
        CA --> RES
        RES --> FFN
        FFN --> FRES
    end

    subgraph Output["Output"]
        FN["Final LayerNorm"]
        OP["Linear Projection\nvision_dim → llm_dim"]
        PT["Projected Tokens\n(batch, num_latents=64, llm_dim)"]
        FN --> OP --> PT
    end

    VT --> NM
    LA --> NL
    FRES -->|"next layer"| NM
    FRES -->|"next layer"| NL
    FRES --> FN

    style Input fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style Layer fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style Output fill:#1e4d4d,stroke:#50c878,stroke-width:3px,color:#fff
    style VT fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style LA fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style NM fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style NL fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style CAT fill:#4a1c4a,stroke:#e066ff,stroke-width:2px,color:#fff
    style CA fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style RES fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style FFN fill:#7b341e,stroke:#fc8181,stroke-width:2px,color:#fff
    style FRES fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style FN fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
    style OP fill:#276749,stroke:#68d391,stroke-width:2px,color:#fff
    style PT fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
</div>

This creates a few advantages:

1) Fewer layers are needed compared to QFormer since there is no dedicated self attention block per layer.

2) Faster convergence since the layers are normed before the attention call

3) This method isn't meant to do this but it can handle videos. The visual tokens are flattened and the resampler compresses the data down into 64 latents. I am unsure how well this would do in practice since this was designed for images.

Similar disadvantages to QFormer since we compress the data we are losing information. The latents are optimized to predict the next token in an interleaved image and text sequence. This is slightly different than QFormer which in the BLIP-2 paper gets a different stage of training. BLIP-2 has three losses in stage 1 that the QFormer queries to understand the images with text. For our experiments this wont matter since we are going to compare these three methods and optimize for the next best token. We can directly compare these methods to each other.

