"""Add SEO front matter (description, topic, tags, image, math) to posts.

Only adds or replaces these keys. Never touches title, date, or categories,
because those control the post URL.

Usage: python scripts/add_front_matter.py
"""

import pathlib
import re

POSTS = pathlib.Path(__file__).resolve().parent.parent / "_posts"

META = {
    "in-the-beginning": dict(
        topic="Introduction",
        tags=["machine-learning", "software-engineering", "career", "teams"],
        description="Why this blog exists: the gap between ML research and real software, why ML work is hard to estimate, and the topics Gradient Rage will cover.",
    ),
    "roller-coaster-tycoon-program-control-flow": dict(
        topic="Software Engineering",
        tags=["software-engineering", "control-flow", "abstraction", "readability", "refactoring"],
        description="Deep abstraction layers make code hard to follow. How to keep program control flow readable, testable, and easy for teammates to change.",
    ),
    "data-imbalances": dict(
        topic="Machine Learning",
        tags=["machine-learning", "imbalanced-data", "smote", "oversampling", "undersampling", "imbalanced-learn"],
        description="Handling imbalanced datasets with undersampling, oversampling, SMOTE, and combined methods, using a credit card fraud dataset and imbalanced-learn.",
        math=True,
    ),
    "object-oriented-programming-4-pillars": dict(
        topic="Software Engineering",
        tags=["oop", "software-engineering", "encapsulation", "inheritance", "polymorphism", "composition"],
        description="The 4 pillars of OOP are not unique to OOP. Encapsulation, inheritance, polymorphism, and abstraction compared with C, composition, and ECS.",
    ),
    "visualizing-loss-gradient": dict(
        topic="Machine Learning",
        tags=["deep-learning", "loss-landscape", "pytorch", "visualization", "training"],
        description="How to visualize the loss landscape of a neural network in PyTorch, compare initialized and trained weights, and track the loss path during training.",
        math=True,
    ),
    "gradient-probing": dict(
        topic="Machine Learning",
        tags=["deep-learning", "gradients", "debugging", "pytorch", "training"],
        description="Debug neural network training by probing gradients: detect vanishing and exploding gradients, measure gradient signal-to-noise ratio and variance.",
    ),
    "sir-we-have-lost-the-processing-step": dict(
        topic="Machine Learning",
        tags=["mlops", "pytorch", "torchscript", "preprocessing", "production", "checkpoints"],
        description="Models break in production when pre- and post-processing drift. How to save processing steps with checkpoints, TorchScript, and baked-in pipelines.",
    ),
    "batch-norm": dict(
        topic="Machine Learning",
        tags=["deep-learning", "batch-normalization", "optimization", "training"],
        description="What batch normalization does, why internal covariate shift is not the real reason it works, and how smoother loss landscapes explain its benefits.",
        math=True,
    ),
    "gpu-programming-in-python-part-1": dict(
        topic="GPU Programming",
        tags=["gpu", "cuda", "python", "cupy", "numba", "performance"],
        description="Run Python on the GPU with CuPy and Numba. Benchmarks of CuPy against NumPy and SciPy, and writing custom CUDA kernels with Numba.",
    ),
    "gpu-programming-in-python-part-2": dict(
        topic="GPU Programming",
        tags=["gpu", "cuda", "python", "cuda-python", "nvrtc", "performance"],
        description="Write and launch CUDA kernels from Python with NVIDIA cuda-python, using both cuda.bindings and cuda.core, with a Gaussian blur and timing results.",
    ),
    "merry-christmas-ai-slop-for-all": dict(
        topic="Opinion",
        tags=["ai", "career", "tech-industry", "layoffs", "software-engineering"],
        description="AI slop, layoffs, and broken interviews in the tech industry, and practical advice for developers and students on how to get through it.",
    ),
    "the-dark-arts-binding-c-to-python": dict(
        topic="GPU Programming",
        tags=["python", "cpp", "c", "pybind11", "ctypes", "cffi", "performance"],
        description="Binding C and C++ to Python with the C API, ctypes, CFFI, SWIG, Boost.Python, and pybind11, with timing results and when each is worth it.",
    ),
    "tracking-models-designing-checkpoints": dict(
        topic="Machine Learning",
        tags=["mlops", "pytorch", "checkpoints", "mlflow", "reproducibility"],
        description="What a good PyTorch checkpoint saves: model, optimizer, scheduler, GradScaler, and RNG states, plus tracking checkpoints with MLflow.",
    ),
    "how-to-create-multi-modal-models-part-1": dict(
        topic="Machine Learning",
        tags=["multi-modal", "llm", "projection-layers", "cross-attention", "vq-gan"],
        description="How multi-modal models understand images and audio: projection layers, cross attention, and VQ-GAN tokens, with their tradeoffs.",
        math=True,
    ),
    "how-to-create-multi-modal-models-part-2": dict(
        topic="Machine Learning",
        tags=["multi-modal", "llm", "q-former", "perceiver-resampler", "pytorch"],
        description="Training an LLM to understand images with an MLP projection, Q-Former, and Perceiver Resampler, with attention maps and inference comparisons.",
    ),
    "good-code-vs-bad-code": dict(
        topic="Software Engineering",
        tags=["software-engineering", "code-quality", "problem-solving", "teams"],
        description="Good code versus bad code is the wrong question. Code is a vehicle for solving problems, shown through API, debugging, and memory layout examples.",
    ),
    "you-are-not-a-10x-engineer": dict(
        topic="Opinion",
        tags=["ai", "agents", "vibe-coding", "productivity", "career"],
        description="AI agents do not make you a 10x engineer. Why vibe-coded software breaks, why the productivity gains are overstated, and why skills still matter.",
    ),
    "explaining-floats": dict(
        topic="Machine Learning",
        tags=["floating-point", "gpu", "mixed-precision", "bf16", "fp8", "fp4"],
        description="IEEE 754 floats explained, then TF32, BF16, FP16, FP8, and FP4 for deep learning: bit layouts, ranges, GPU support, and when to use each.",
    ),
    "complex-number-and-deep-learning-part1": dict(
        topic="Signal Processing",
        tags=["complex-numbers", "deep-learning", "wirtinger-calculus", "signal-processing", "math"],
        description="Complex numbers for deep learning, with interactive visuals: polar form, rotation, holomorphic functions, and Wirtinger calculus for backprop.",
        math=True,
    ),
    "complex-number-and-deep-learning-part2": dict(
        topic="Signal Processing",
        tags=["complex-numbers", "deep-learning", "rf", "modulation-classification", "rf-fingerprinting"],
        description="Complex-valued versus real-valued neural networks on modulation classification and RF fingerprinting, with results and when complex layers help.",
    ),
    "read-the-code": dict(
        topic="Opinion",
        tags=["ai", "agents", "vibe-coding", "code-review", "software-engineering"],
        description="Should you read the code your AI agents write? Yes. Why the hard part of software was never only the code, and why agents still need review.",
    ),
    "spooky-productivity": dict(
        topic="Opinion",
        tags=["ai", "productivity", "code-review", "burnout", "software-engineering"],
        description="AI productivity pressure floods teams with unreviewed merge requests. How to push back and spend the time on deeper, better work instead.",
    ),
    "rubber-duck-documentation": dict(
        topic="Software Engineering",
        tags=["documentation", "llm", "agents", "software-engineering", "cli"],
        description="Use LLMs as a rubber duck to write short, direct documentation for programs and CLIs that both teammates and AI agents can use.",
    ),
}

MANAGED = ("description", "topic", "tags", "image", "math")


def slug_of(path):
    return re.sub(r"^\d{4}-\d{1,2}-\d{1,2}-", "", path.stem)


def yaml_str(value):
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def update(path):
    slug = slug_of(path)
    meta = META.get(slug)
    if meta is None:
        print(f"skip (no metadata): {path.name}")
        return
    text = path.read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        raise SystemExit(f"no front matter: {path}")
    lines = [l for l in match.group(1).split("\n") if l.split(":", 1)[0].strip() not in MANAGED]
    lines.append(f"topic: {meta['topic']}")
    lines.append("tags: [" + ", ".join(meta["tags"]) + "]")
    lines.append(f"description: {yaml_str(meta['description'])}")
    lines.append(f"image: /assets/images/social/{slug}.png")
    if meta.get("math"):
        lines.append("math: true")
    path.write_text("---\n" + "\n".join(lines) + "\n---\n" + text[match.end():])
    print(f"updated: {path.name}")


if __name__ == "__main__":
    for post in sorted(POSTS.glob("*")):
        if post.suffix in (".md", ".markdown"):
            update(post)
