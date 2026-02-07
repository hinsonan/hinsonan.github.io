---
layout: post
title: "How to Create Multi Modal Models Part 1"
date: 2026-02-07
categories: ML
---

Remember in Bloodborne how your character gains insight and you can start to see all these "hidden" creatures that were there the whole time? Well, in some ways those creatures were projected into a space in which you could begin to comprehend them. It may be a little stretch, but this is how you can make models "understand" multiple data modalities. Multi-modal models are very powerful and you need to understand how they work and how to create them.

## Multi Modal Understanding

How do some of these modern models process data of different modalities? How can I give an LLM an image or audio and it "understand" those data types? There are a few ways this can be done, and each technique has multiple different methods you can perform to get a model to understand multiple domains.

Three main ways this can be done are

1) Projection Layers

2) Cross Attention

3) VQ-GAN (Image Tokens, Audio Tokens, etc...)

There are always new research papers coming out that try new things, but these are some of the better established methods.

## Projection Layers

This was the first large-scale successful method, and it's very simple and intuitive. This became popular from [LLaVA](https://llava-vl.github.io/). Honestly, without their open-source contributions, I do not think multi-modal models would be where they are today. Many researchers still use their methods and insights to create these models.

### How it works

You can take a pre-trained vision model like CLIP and a text-only model like Qwen/Llama, then "combine" them together to have a model that can understand images. The image is projected into the text embedding space. The reason for this is the LLM has built powerful encodings and is trained with a certain fixed-size embedding space.

The LLM can only understand what is in this embedding space. The image you have must be projected into this embedding space so the model can "see" it. Essentially, the LLM only talks a certain language and you need to get the image to be in this "language."

<div class="mermaid">
flowchart TB
    subgraph Step1["1. Start with an Image"]
        IMG["Your Image<br/>(Cat, Chart)"]
    end
    
    subgraph Step2["2. Vision Model Understands the Image"]
        VISION["Vision Encoder<br/>(like CLIP)<br/><br/>Converts image into<br/>numbers the computer understands"]
    end
    
    subgraph Step3["3. The Magic: Projection Layer"]
        PROJ["Projection Layer<br/><br/>Translates from<br/>'Vision Language' → 'LLM Language'"]
    end
    
    subgraph Step4["4. LLM Processes Everything"]
        LLM["Large Language Model<br/>(like Qwen/Llama)<br/><br/>Sees both:<br/>• Your text prompt<br/>• The projected image"]
    end
    
    subgraph Step5["5. Output"]
        OUT["Response<br/>'I see a fluffy cat!'"]
    end
    
    IMG --> VISION
    VISION -->|"Image features<br/>(just numbers!)"| PROJ
    PROJ -->|"Now in LLM's<br/>language"| LLM
    LLM --> OUT
    
    style Step1 fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style Step2 fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style Step3 fill:#1e4d4d,stroke:#50c878,stroke-width:4px,color:#fff
    style Step4 fill:#4a1c4a,stroke:#e066ff,stroke-width:3px,color:#fff
    style Step5 fill:#2d5a2d,stroke:#7fff7f,stroke-width:3px,color:#fff
    style IMG fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style VISION fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style PROJ fill:#234e52,stroke:#4fd1c5,stroke-width:3px,color:#fff
    style LLM fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style OUT fill:#276749,stroke:#68d391,stroke-width:2px,color:#fff
</div>

When you project the image into the same shape and embedding space of the LLM, then the LLM can learn what these image tokens mean. Let's look at a more detailed explanation.

### How Text and Images Combine

The way this works is that there is generally an `<image>` token in the sentence that gets replaced with the real tokens. For example `Describe this image: <image>`. Each token would be replaced with the embedding size of the LLM and the image token will be replaced with all of its tokens.

<div class="mermaid">
flowchart TB
    subgraph UserPrompt["1. User Prompt with Placeholder"]
        PROMPT["Describe this image: &lt;image&gt;<br/><br/>The LLM sees a special marker<br/> where the image will go"]
    end
    
    subgraph Tokenization["2. Tokenize the Sentence"]
        TOKENS_RAW["Raw Tokens<br/><br/>[Describe, this, image, :, &lt;image&gt;]<br/>Shape: [5 tokens]"]
    end
    
    subgraph ImagePath["3. Image Processing Pipeline"]
        direction TB
        IMG_INPUT["Input Image<br/>Shape: [224, 224, 3]"]
        VISION_ENC["Vision Encoder<br/>ViT/CLIP"]
        IMG_FEATURES["Image Features<br/>Shape: [576, 768]"]
        PROJ["Projection Layer<br/>Linear(768 → 4096)"]
        PROJ_RESULT["Projected Features<br/>Shape: [576, 4096]<br/>576 image tokens"]
    end
    
    subgraph Expansion["4. The Expansion Step"]
        direction TB
        BEFORE["Before:<br/>[Describe, this, image, :, PLACEHOLDER]<br/>5 tokens total"]
        ARROW["Replace &lt;image&gt; with 576<br/>projected image tokens"]
        AFTER["After:<br/>[Describe, this, image, :, IMG₀, IMG₁, ..., IMG₅₇₅]<br/>580 tokens total"]
    end
    
    subgraph LLMView["5. What the LLM Sees"]
        direction TB
        LLM_INPUT["Input to LLM:<br/>A single sequence of 580 tokens<br/><br/>• Text tokens (4): 'Describe this image :'<br/>• Image tokens (576): Visual information<br/><br/>All in same embedding space [580, 4096]"]
        ATTENTION["Self-Attention Across<br/>All 580 Tokens<br/><br/>Each token can 'look at' every other token<br/>Text ↔ Text, Text ↔ Image, Image ↔ Image"]
        PROCESSING["LLM Processes:<br/>'Describe' looks at image patches<br/>to understand what to describe"]
    end
    
    subgraph Output["6. Generate Response"]
        RESPONSE["The cat is orange<br/>and sleeping on a couch"]
    end
    
    PROMPT --> TOKENS_RAW
    IMG_INPUT --> VISION_ENC
    VISION_ENC --> IMG_FEATURES
    IMG_FEATURES --> PROJ
    PROJ --> PROJ_RESULT
    
    TOKENS_RAW --> BEFORE
    PROJ_RESULT --> ARROW
    ARROW --> AFTER
    
    AFTER --> LLM_INPUT
    LLM_INPUT --> ATTENTION
    ATTENTION --> PROCESSING
    PROCESSING --> RESPONSE
    
    style UserPrompt fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style Tokenization fill:#2d5a2d,stroke:#7fff7f,stroke-width:3px,color:#fff
    style ImagePath fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style Expansion fill:#1e4d4d,stroke:#50c878,stroke-width:4px,color:#fff
    style LLMView fill:#4a1c4a,stroke:#e066ff,stroke-width:3px,color:#fff
    style Output fill:#702459,stroke:#f687b3,stroke-width:3px,color:#fff
    
    style PROMPT fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style TOKENS_RAW fill:#276749,stroke:#68d391,stroke-width:2px,color:#fff
    style IMG_INPUT fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style VISION_ENC fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style IMG_FEATURES fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style PROJ fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style PROJ_RESULT fill:#234e52,stroke:#4fd1c5,stroke-width:3px,color:#fff
    style BEFORE fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style ARROW fill:#234e52,stroke:#4fd1c5,stroke-width:3px,color:#fff
    style AFTER fill:#234e52,stroke:#4fd1c5,stroke-width:3px,color:#fff
    style LLM_INPUT fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style ATTENTION fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style PROCESSING fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style RESPONSE fill:#2d5a2d,stroke:#68d391,stroke-width:3px,color:#fff
</div>

**Understanding Matrix Shapes:**
- `[5]` = 5 text tokens (including the `<image>` placeholder)
- `[224, 224, 3]` = Standard image size (224×224 pixels, 3 color channels: RGB)
- `[576, 768]` = 576 image patches, each with 768 features (from vision encoder)
- `[576, 4096]` = After projection: same 576 patches, now with 4096 dimensions to match the LLM
- `[580, 4096]` = Final sequence: 4 text tokens + 576 image tokens = 580 total tokens, each with 4096 dimensions

The `<image>` token acts as a placeholder that gets expanded into 576 actual image tokens. This is how models can understand different modalities like images. You have to find a common space to talk within. Each LLM has a different size embedding space so you have to project to whatever size that is.

## Cross Attention

Cross Attention works a bit differently. Instead of packing all the image tokens into the context (which can overload or saturate the context), you can perform attention across the image and pull out the most relevant information to put into the prompt. It is like 20 questions where you are learning what questions or features to take from the image and use those learned tokens in the prompt.

Models like [Flamingo](https://arxiv.org/pdf/2204.14198) use this technique.

### The Cross Attention Formula

$$\text{CrossAttention}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}}) = \text{softmax}\left(\frac{Q_{\text{text}} K_{\text{image}}^T}{\sqrt{d_k}}\right) V_{\text{image}}$$

<div class="mermaid">
flowchart LR
    subgraph Input["Input"]
        direction TB
        TEXT["Text: What color is the cat?"]
        IMG["Image: [Cat Photo]"]
    end
    
    subgraph CrossAttention["Cross Attention"]
        direction TB
        STEP1["1. Text generates Queries<br/>'Where is the cat?'"]
        STEP2["2. Image provides Keys & Values<br/>Each patch says 'I contain X'"]
        STEP3["3. Match Queries to Keys<br/>Find which patches have the cat"]
        STEP4["4. Retrieve Values<br/>Get visual info from those patches"]
    end
    
    subgraph Output["Output"]
        ENRICHED["Enriched Text:<br/>'cat' now knows it's orange"]
    end
    
    TEXT --> STEP1
    IMG --> STEP2
    STEP1 --> STEP3
    STEP2 --> STEP3
    STEP3 --> STEP4
    STEP4 --> ENRICHED
    
    style Input fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style CrossAttention fill:#1e4d4d,stroke:#50c878,stroke-width:4px,color:#fff
    style Output fill:#702459,stroke:#f687b3,stroke-width:3px,color:#fff
    
    style TEXT fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style IMG fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style STEP1 fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style STEP2 fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style STEP3 fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style STEP4 fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style ENRICHED fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
</div>

### Q, K, V in Cross Attention

The three components work like a search system or dictionary lookup:

<div class="mermaid">
flowchart LR
    subgraph Query["Query (Q)"]
        Q_TEXT["From: Text<br/><br/>'What color is the cat?'"]
        Q_DESC["Think of Q as a<br/>SEARCH QUESTION<br/><br/>Each word asks:<br/>'Where can I find X?'"]
    end
    
    subgraph Key["Key (K)"]
        K_TEXT["From: Image<br/><br/>Image patches describe<br/>what they contain"]
        K_DESC["Think of K as<br/>LABELS/TAGS<br/><br/>Each patch says:<br/>'I contain cat/orange/fur'"]
    end
    
    subgraph Value["Value (V)"]
        V_TEXT["From: Image<br/><br/>Image patches actual content"]
        V_DESC["Think of V as<br/>THE ACTUAL CONTENT<br/><br/>Each patch holds:<br/>'Here's what I look like'"]
    end
    
    subgraph Matching["Matching Process"]
        MATCH["Q matches with K<br/><br/>Text 'cat' searches<br/>→ Finds patches tagged 'cat'<br/><br/>Result: attention weights"]
    end
    
    subgraph Retrieval["Retrieval"]
        RETRIEVE["Retrieve V using weights<br/><br/>Get the actual visual<br/>content from matched<br/>patches"]
    end
    
    Q_TEXT --> Q_DESC
    K_TEXT --> K_DESC
    V_TEXT --> V_DESC
    
    Q_DESC --> MATCH
    K_DESC --> MATCH
    MATCH --> RETRIEVE
    V_DESC --> RETRIEVE
    
    style Query fill:#2d5a2d,stroke:#7fff7f,stroke-width:3px,color:#fff
    style Key fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style Value fill:#702459,stroke:#f687b3,stroke-width:3px,color:#fff
    style Matching fill:#1e4d4d,stroke:#50c878,stroke-width:4px,color:#fff
    style Retrieval fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    
    style Q_TEXT fill:#276749,stroke:#68d391,stroke-width:2px,color:#fff
    style Q_DESC fill:#276749,stroke:#68d391,stroke-width:2px,color:#fff
    style K_TEXT fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style K_DESC fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style V_TEXT fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style V_DESC fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style MATCH fill:#234e52,stroke:#4fd1c5,stroke-width:3px,color:#fff
    style RETRIEVE fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
</div>

**Components:**
- **Query (Q)** comes from your **text** → "What am I looking for?"
- **Key (K)** comes from the **image** → "What do I contain?"
- **Value (V)** comes from the **image** → "What information do I provide?"
- **$d_k$** = dimension of the keys (used for scaling to prevent softmax saturation)

The softmax function will create a heatmap over the image that relates to the words that you input to grab the most relevant parts of the image related to the text query.

### Token Efficiency Comparison

Here is a table that breaks down the number of tokens used by projection vs cross attention.

| Aspect | Projection Layers | Cross Attention |
|--------|------------------|-----------------|
| **Text Tokens** | 4 tokens<br/>"Describe this image :" | 4 tokens<br/>"Describe this image :" |
| **Image Processing** | All 576 image patches<br/>projected into LLM space | 576 patches stay in<br/>image encoder space |
| **Tokens in LLM Context** | 4 + 576 = **580 tokens** | 4 + 64 = **68 tokens**<br/>(learned query tokens) |
| **Context Usage** | High - full image in context | Low - only compressed queries |
| **Information Flow** | One-time projection,<br/>then self-attention | Continuous cross-attention<br/>at each layer |
| **Flexibility** | Fixed once projected | Dynamic - queries adapt<br/>per layer |

**Main Differences:**
- **Projection layers**: Convert image features once to match LLM embeddings, then process together
- **Cross attention**: Text and image features interact at all layers through attention

This makes cross attention more flexible and helps manage context since you only learn a fixed number of tokens, which is much less than projecting the whole image into the prompt. The trade-off is cross-attention is computationally expensive and it can take longer to train. Due to how cross attention works, it is not the best method for **OCR** tasks since you are only querying a handful of the image tokens. Depending on your use case, each of these methods has its place.

## VQ-GAN (Image Tokens, Audio Tokens, etc...)

VQ-GAN approaches this problem differently and makes it so the images or other modalities become a new learned vocabulary that the LLM can understand. This is like an LLM learning a new "language".

### How It Works

The VQ-GAN is like a lookup for image patches to find which of the new vocab tokens it is most similar to.

<div class="mermaid">
flowchart TB
    subgraph Input["Input: Photo of a Cat"]
        RAW["Raw Image<br/> 512 x 512 pixels"]
    end
    
    subgraph Tokenizer["VQ-GAN Tokenizer"]
        direction TB
        ENCODE["Encoder compresses image<br/>into latent representations"]
        CODEBOOK["Codebook Lookup<br/><br/>A learned dictionary of<br/>8,192 visual patterns<br/><br/>Each pattern is like a<br/>'visual word'"]
        TOKENS["Discrete Tokens<br/><br/>1,024 integer tokens<br/>representing the image"]
    end
    
    subgraph Unified["Unified Representation"]
        COMBINED["Both text and images<br/>are just tokens!<br/><br/>Text tokens: [The, cat, sits, ...]<br/>Image tokens: [IMG_42, IMG_105, ...]<br/><br/>All in the same vocabulary"]
    end
    
    subgraph Transformer["Single Transformer"]
        PROCESS["One model processes<br/>all token types<br/><br/>No separate encoders<br/>No projection layers<br/>No cross attention"]
    end
    
    subgraph Output["Output"]
        RESULT["Can generate both<br/>text & images"]
    end
    
    RAW --> ENCODE
    ENCODE --> CODEBOOK
    CODEBOOK --> TOKENS
    TOKENS --> COMBINED
    COMBINED --> PROCESS
    PROCESS --> RESULT
    
    style Input fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style Tokenizer fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style Unified fill:#1e4d4d,stroke:#50c878,stroke-width:4px,color:#fff
    style Transformer fill:#4a1c4a,stroke:#e066ff,stroke-width:3px,color:#fff
    style Output fill:#702459,stroke:#f687b3,stroke-width:3px,color:#fff
    
    style RAW fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style ENCODE fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style CODEBOOK fill:#975a16,stroke:#f6ad55,stroke-width:3px,color:#fff
    style TOKENS fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style COMBINED fill:#234e52,stroke:#4fd1c5,stroke-width:3px,color:#fff
    style PROCESS fill:#702459,stroke:#f687b3,stroke-width:2px,color:#fff
    style RESULT fill:#2d5a2d,stroke:#68d391,stroke-width:3px,color:#fff
</div>

### What is a Codebook?

A codebook is a fixed set of tokens that are learned for images or other modalities. It is essentially like adding a language pack to the LLM or adding extra vocabulary to the model.

In short, it works like this:
- **8,192 visual patterns** (Generally 8-16k depending on model and this is similar to a visual alphabet)
- Each image patch gets matched to the closest pattern
- Results in discrete integers (tokens) the LLM can process

### How Codebook Quantization Works

<div class="mermaid">
flowchart TB
    subgraph Image["Input: Simple Image"]
        IMG["[Image of a sunset]<br/><br/>Top: Blue sky<br/> Middle: Orange clouds<br/> Bottom: Dark ground"]
    end
    
    subgraph EncodeStep["Step 1: Encode Image"]
        ENCODE["Encoder CNN processes image<br/>Breaks into 32x32 patches<br/><br/>Each patch becomes a<br/>continuous vector"]
        VECTORS["Patch Vectors:<br/>Patch 1: [0.3, -0.8, 0.5, ...]<br/>Patch 2: [0.9, 0.2, -0.1, ...]<br/>Patch 3: [-0.5, -0.9, 0.3, ...]<br/>(continuous numbers)"]
    end
    
    subgraph CodebookStep["Step 2: Match to Codebook"]
        CODEBOOK["Codebook:<br/>A library of 8,192 patterns<br/><br/>ID 42: Blue sky pattern<br/>ID 128: Orange cloud pattern<br/>ID 256: Dark ground pattern<br/>ID 512: Green grass pattern<br/>... (many more)"]
        MATCHING["For each patch,<br/>find closest match:<br/><br/>Patch 1 → ID 42<br/>Patch 2 → ID 128<br/>Patch 3 → ID 256"]
    end
    
    subgraph TokenStep["Step 3: Discrete Tokens"]
        TOKENS["Result: Integer Tokens<br/><br/>[42, 128, 256, ...]<br/><br/>Just like text tokens!<br/>The LLM sees:<br/>['the', 'sun', 'sets', IMG_42, IMG_128, IMG_256]"]
    end
    
    IMG --> ENCODE
    ENCODE --> VECTORS
    VECTORS --> MATCHING
    CODEBOOK --> MATCHING
    MATCHING --> TOKENS
    
    style Image fill:#1e3a5f,stroke:#4a90e2,stroke-width:3px,color:#fff
    style EncodeStep fill:#744210,stroke:#f5a623,stroke-width:3px,color:#fff
    style CodebookStep fill:#1e4d4d,stroke:#50c878,stroke-width:4px,color:#fff
    style TokenStep fill:#702459,stroke:#f687b3,stroke-width:3px,color:#fff
    
    style IMG fill:#2c5282,stroke:#63b3ed,stroke-width:2px,color:#fff
    style ENCODE fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style VECTORS fill:#975a16,stroke:#f6ad55,stroke-width:2px,color:#fff
    style CODEBOOK fill:#234e52,stroke:#4fd1c5,stroke-width:3px,color:#fff
    style MATCHING fill:#234e52,stroke:#4fd1c5,stroke-width:2px,color:#fff
    style TOKENS fill:#702459,stroke:#f687b3,stroke-width:3px,color:#fff
</div>

This method turns the image into numbers that the LLM can learn. This has a big advantage in that this can also be used to generate text-to-image. You can use this approach and generate images since the image is just a number and the model can predict the next image token when trained this way. Models like [Chameleon](https://arxiv.org/pdf/2405.09818) and [Janus-Pro](https://arxiv.org/pdf/2501.17811) use this technique.

### Pros/Cons of Codebooks

Codebooks are very powerful, and they provide a more "natural" tokenization where the LLM only has to learn this new vocabulary. As always, there is no free lunch, and this method has its pros/cons.

**Pros:**

1. **Unified Representation**
   - Images become discrete integer tokens just like text
   - Enables both understanding AND generation (text-to-image) in a single model

2. **Computational Efficiency**
   - Discrete tokenization is more efficient
   - VQ can compress weight and KV cache tensors with greater ratios

3. **Strong Compression with Quality**
   - Learns a codebook of context-rich visual parts with high perceptual quality

**Cons:**

1. **Information Loss**
   - Quantization discards information
   - Not ideal for OCR or tasks requiring pixel-perfect accuracy

2. **Training Challenges**
   - Codebook collapse issues during training
   - Unstable gradient estimation through the quantization bottleneck

### Image Generation with VQ-GAN

Using a VQ-GAN, you can now generate images with this model and not just text. This makes this method very powerful. This works by having an auto-regressive codebook to look up. To generate an image, you are simply generating image codes/tokens. For example, image token `<IMG_100>` may represent blue skies and the next token may be a token that corresponds to clouds in the sky. These tokens will be decoded into patches and a CNN or some other model may scale them back to a certain shape.

## When to Use Projection/Cross Attention/VQ-GAN

Depending on your goals, each of these methods can be used to great success. Here are some tasks where each method can succeed or fail.

| Method | Excels At | Struggles With |
|--------|-----------|----------------|
| **Projection Layers** | • OCR and text-heavy documents<br/>• Dense document understanding<br/>• Tasks requiring fine-grained visual details<br/>• General vision-language understanding | • Very long context windows (uses 576+ tokens per image)<br/>• Memory-constrained environments<br/>• Batch processing many images<br/>• Real-time applications with limited compute |
| **Cross Attention** | • General visual question answering<br/>• Context-efficient multimodal understanding<br/>• Long-form visual reasoning<br/>• Flexible content querying<br/>• Processing multiple images in limited context | • OCR and fine-grained text recognition<br/>• Tasks requiring complete spatial information<br/>• Pixel-level precision tasks<br/>• Dense document layouts<br/>• Detailed chart/diagram analysis |
| **VQ-GAN / Codebook** | • Image generation (text-to-image)<br/>• Unified understanding + generation models<br/>• Creative visual tasks<br/>• Style transfer and manipulation<br/>• Learning cross-modal representations | • OCR and precise text recognition<br/>• Fine-grained visual details<br/>• Tasks requiring pixel-perfect accuracy |

### Tokenization Comparison

| Aspect | Projection Layers | Cross Attention | VQ-GAN / Codebook |
|--------|-------------------|-----------------|---------------------|
| **Representation Type** | Continuous vectors | Continuous vectors | Discrete integers |
| **Tokens per Image** | ~576 (all patches) | ~64 (learned queries) | ~1,024 (codebook indices) |
| **Information Preservation** | High - all patch info retained | Selective - only queried info | Lossy - quantization discards detail |
| **How Image Enters LLM** | Projected tokens concatenated into prompt | Text queries image via attention at each layer | Image tokens added to vocabulary, treated like text |
| **Requires Separate Vision Encoder** | Yes (CLIP, SigLIP, etc.) | Yes (CLIP, SigLIP, etc.) | No - uses its own trained tokenizer |
| **Can Generate Images** | No | No | Yes |
| **Training Complexity** | Low - just train projection layer | Medium - add cross-attention layers | High - codebook collapse risk, unstable gradients |

Cross attention gives you the best results for image understanding compared to VQ-GAN. Context management is much easier with cross attention, and you can query against the image to get a smaller fixed-size set of tokens. VQ-GANs are great for image generation, unified vocabulary, and other types of model outputs.

## Conclusion

Now we know how models can learn to accept different modalities than just the one modality they were originally trained on or meant to understand.

**Multi Modal Recap**: 
- Projection layers preserve the most visual detail and are best for OCR and document understanding.
- Cross attention gives you context efficiency by querying only the most relevant image features.
- VQ-GAN unlocks image generation by turning images into discrete tokens the LLM can predict.

Each method trades off between information preservation, context efficiency, and generation capability.

In the next article, we will use some of these techniques to train our own model to understand different domains.