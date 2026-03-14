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

    subgraph MLP["Two-Layer MLP Projection - LLaVA-1.5 Style"]
        L1["Linear Layer 1<br/>vision_dim to llm_dim"]
        GELU["GELU Activation"]
        L2["Linear Layer 2<br/>llm_dim to llm_dim"]
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

    subgraph Layer["Transformer Layer x12"]
        SA["Self-Attention<br/>queries attend to queries"]
        SN["LayerNorm"]
        CA["Cross-Attention - every other layer<br/>queries attend to visual tokens"]
        CN["LayerNorm"]
        FFN["FFN - Linear to GELU to Linear<br/>vision_dim to vision_dim x4 to vision_dim"]
        FN["LayerNorm"]

        SA --> SN --> CA --> CN --> FFN --> FN
    end

    subgraph Output["Output"]
        OP["Linear Projection<br/>vision_dim to llm_dim"]
        PT["Projected Tokens<br/>batch, num_queries=32, llm_dim"]
        OP --> PT
    end

    VT -->|"keys and values"| CA
    QT --> SA
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
        VT["Visual Tokens<br/>(batch, num_patches, vision_dim)"]
        LA["Learnable Latents<br/>(1, num_latents=64, vision_dim)"]
    end

    subgraph Layer["Perceiver Layer x2"]
        NM["LayerNorm - visual tokens"]
        NL["LayerNorm - latents"]
        CAT["Concat<br/>normed visual tokens + normed latents<br/>keys and values"]
        CA["Cross-Attention<br/>Q = normed latents<br/>K and V = concat above"]
        RES["Residual Add<br/>latents = latents + attn_out"]
        FFNORM["LayerNorm - pre FFN norm"]
        FFN["FFN - Linear to GELU to Linear<br/>vision_dim to vision_dim x4 to vision_dim"]
        FRES["Residual Add<br/>latents = latents + ffn_out"]

        NM --> CAT
        NL --> CA
        NL --> CAT
        CAT --> CA
        CA --> RES
        RES --> FFNORM
        FFNORM --> FFN
        FFN --> FRES
    end

    subgraph Output["Output"]
        FN["Final LayerNorm"]
        OP["Linear Projection<br/>vision_dim to llm_dim"]
        PT["Projected Tokens<br/>batch, num_latents=64, llm_dim"]
        FN --> OP --> PT
    end

    VT --> NM
    LA --> NL
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
    style FFNORM fill:#553c10,stroke:#f6e05e,stroke-width:2px,color:#fff
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

## Setting Up The Experiment

<div style="background:#3d2b00;border-left:4px solid #f5a623;padding:12px 16px;border-radius:4px;color:#fbd38d;margin:16px 0;">
  <strong>⚠ Note:</strong> The purpose of these experiments is to show how the model can learn to understand the image modality. This is not a rigorous testing about which of these three methods is the best if you limit all methods to the same number of output tokens. QFormer is also at a disadvantage since we are not doing staged training like the BLIP-2 paper. The goal of this experiment is for you to get an idea of how these projection work and see how it effects the model.
</div>

We want to learn more about how these different projection layers effect the model. We want to know some of the tradeoffs and when to use certain projection methods.

We are going to be using the [coco-karpathy](https://huggingface.co/datasets/yerevann/coco-karpathy) dataset. This is an image caption pair dataset. Due to time training will only be 10 epochs using 1000 train and 1000 validation examples. One of the benefits of projection layers is they are faster to train so we will see how this does.

For our vision model we will be using [SigLIP](https://huggingface.co/google/siglip-base-patch16-224). This model is similar to CLIP and it will output 16x16 patches for a 224 pixel image that gives you 196 patches.

For our LLM we will be using [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

The projection layer's job is to bridge the gap between these two embedding spaces.

<div class="mermaid">
flowchart TB
    IMG["SigLIP Vision Encoder<br/>(batch, 196, 768)"]
    PROJ["Projection Layer<br/>768 → 896"]
    LLM["Qwen 0.5B<br/>(batch, tokens, 896)"]

    IMG -->|"196 × 768-dim tokens"| PROJ
    PROJ -->|"896-dim tokens"| LLM

    style IMG fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style PROJ fill:#7b341e,stroke:#fc8181,stroke-width:2px,color:#fff
    style LLM fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
</div>

Now we have to train the projection layer to project the SigLIP embeddings into the same embedding space as the Qwen Model.

We will be using the **CIDEr** (Consensus-based Image Description Evaluation) metric to determine how the model is performing. This metric is designed for image captioning tasks.

Another metric we will use is **METEOR** (Metric for Evaluation of Translation with Explicit ORdering). It is meant to be used for translation task but it does well for image captioning. It compares the output to one or more references with different priorities.



### Initial Untrained Inference

Let's see how the model does with no training.

![coco_2](/assets/images/projection_layer_images/untrained/coco_2.png)
![coco_15](/assets/images/projection_layer_images/untrained/coco_15.png)
![coco_36](/assets/images/projection_layer_images/untrained/coco_36.png)
![coco_53](/assets/images/projection_layer_images/untrained/coco_53.png)
![coco_57](/assets/images/projection_layer_images/untrained/coco_57.png)
![coco_60](/assets/images/projection_layer_images/untrained/coco_60.png)

As you can see this model has no idea how to process or understand images.

## Running Training and Inference Results

### Results Summary

All three models were trained for 10 epochs on 1000 COCO training samples and evaluated on 1000 validation samples.

| Model | Best CIDEr | Final Val Loss | Final CIDEr | Final METEOR | Output Tokens |
|-------|:----------:|:--------------:|:-----------:|:------------:|:-------------:|
| MLP | 1.3100 | 3.2366 | 1.1871 | 0.2052 | 196 |
| Perceiver | **1.4604** | 3.4482 | **1.4551** | **0.2320** | 64 |
| Q-Former | 0.7789 | 2.8455 | 0.7424 | 0.1895 | 32 |

### Inference Comparison

GT = ground truth caption. All six validation images shown.

| Image | Ground Truth | MLP | Perceiver | Q-Former |
|-------|-------------|-----|-----------|----------|
| Boy with umbrella and cattle | A child holding a flowered umbrella and petting a yak. | A boy feeding a cow. | A man is holding the grass that a horse has been bred on. | A bunch of people with a large umbrella standing next to an animal holding wa... |
| Galley kitchen | A galley kitchen with cabinets and appliances on both sides. | A kitchen with a bowl and stove on it. | A small kitchen with a refrigerator and stove in it. | A kitchen with an open stove and some utensils on it. |
| Girl holding a kitten | A young girl is holding a small cat. | A little girl is holding a baby with one hand, the other hand has an elephant. | A girl is keeping her hand between a puppy and her face. | A pink, squawzy ball being held up with a cloth. |
| Bathroom with toilet | The toilet is near the door in the bathroom. | The bathroom has a toilet and an ashtray next to it. | A bathroom with a toilet in the middle and black grout on the walls. | A bathroom containing an toilet and a sink. |
| Steel bathroom | A photo of a bathroom made of steel. | A bathroom with sinks and a bowl in it. | A kitchen has multiple small structures on its walls. | A bathroom with a mirror and a white table with some clothes on it. |
| Woman on bicycle | A woman rides a bicycle on a road next to the median. | A person on a bike sitting by the side of a road. | A person is walking on a sidewalk with bikes behind them. | woman with umbrella on top of car |

Each training took a few hours. While training both the vision encoder and llm were frozen so only the projection layer was trained. You can see that even with this small amount of training we have a model that is beginning the understand images.

Let's take a look at how the QFormer and Resampler "see" the image with their attention scores.

![attention_grids](/assets/images/projection_layer_images/trained_attention_grids.png)

![attention_grids_gradients](/assets/images/projection_layer_images/trained_attention_model_comparison.png)

The QFormer seems to be spread out throughout the whole image with some hotspots and the Resampler seems to lock into certain areas.

What's also neat is you can see the different queries and latents in the projection layer that focus on different areas.

![queries](/assets/images/projection_layer_images/trained_attention_qformer_specialization.png)

![resampler](/assets/images/projection_layer_images/trained_attention_perceiver_specialization.png)

This can show you how your projection layer is focusing on certain aspects of the image. Now this training run was very small so all these results are not mature or conclusive. All this shows is that in a few hours we have started to add a new modality to a model.

## Conclusion

We have taken an language model that can not understand images and in just a few hours it has started to be able to process images. If you wanted you can run the training for longer and get much better results.

Multi modal models will continue to grow and we will see them grow into other domains. Music, audio, video, electrical signals, and much more will become very popular domains that models will learn to process. I hope this peaks your curiosity about multi modal models.

[Training Code](https://github.com/hinsonan/hinsonan.github.io/blob/master/code_examples/code_examples/projection_layers)