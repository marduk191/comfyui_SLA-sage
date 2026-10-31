# ComfyUI SLA-sage

**Sparse-Linear Attention (SLA) nodes for ComfyUI.** These nodes swap the self-attention in diffusion models for [SLA](https://github.com/thu-ml/SLA) block-sparse kernels to speed up inference on long sequences (video and high-resolution image models).

## 🌟 Features

- **SLA Attention node**: SLA Triton kernels with configurable sparsity and query block size
- **SageSLA Attention node**: quantized SageSLA kernels from SpargeAttn, falling back to standard SLA when SpargeAttn isn't installed
- **Works across model families**: hooks ComfyUI's `optimized_attention` override, so it applies to Flux, Wan, HunyuanVideo, Qwen-Image, SD3, SDXL and other models that use ComfyUI's attention
- **Safe fallbacks**: calls the kernels can't handle run on ComfyUI's regular attention instead of erroring
- **Step window**: limit sparse attention to part of the sampling schedule with `start_percent` / `end_percent`

## 📋 What is SLA?

SLA (Sparse-Linear Attention) splits attention into a block-sparse part and a linear part. For each block of queries it computes exact attention against the top-k most relevant key blocks and handles the remaining blocks with cheap linear attention. Skipping most of the key blocks is where the speedup comes from.

> **Note on weights:** the linear branch's projection (`proj_l`) is learned during SLA fine-tuning and is zero-initialised. With ordinary (non-SLA-fine-tuned) checkpoints, the node therefore runs pure top-k block-sparse attention. Quality depends on `topk` and on how well the model tolerates sparsity, so compare against the unpatched model.

## 🔧 Installation

### Method 1: Comfy Registry / ComfyUI Manager (Recommended)

Search for **SLA-sage** in ComfyUI Manager, or install with comfy-cli:

```bash
comfy node install comfyui-sla-sage
```

### Method 2: Manual Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/marduk191/comfyui_SLA-sage.git
   ```

3. Install dependencies:
   ```bash
   cd comfyui_SLA-sage
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Requirements

- A recent ComfyUI (tested on ComfyUI 0.43.0)
- An NVIDIA CUDA GPU (Ampere or newer recommended)
- PyTorch >= 2.7 and Triton >= 3.3 (`triton-windows` on Windows)
- [SLA](https://github.com/marduk191/SLA), installed by `requirements.txt`
- Optional: [SpargeAttn](https://github.com/marduk191/SpargeAttn) for the SageSLA kernels

### Optional: SageSLA (Optimized) Installation

For the SageSLA kernels, install SpargeAttn (needs a CUDA toolchain):

```bash
pip install git+https://github.com/marduk191/SpargeAttn.git --no-build-isolation
```

On Windows you can use the pre-built wheels from [woct0rdho/SpargeAttn](https://github.com/woct0rdho/SpargeAttn/releases). The node includes a compatibility shim for their architecture-specific modules.

Without SpargeAttn, the SageSLA node falls back to standard SLA.

## 📖 Usage

```
Load Diffusion Model / Checkpoint → SLA Attention → KSampler → VAE Decode → Save
```

Connect your MODEL to an SLA node and feed the patched model to the sampler. A ready-made workflow is in [`example_workflows/example_workflow.json`](example_workflows/example_workflow.json).

### SLA Attention

| Input | Description |
|---|---|
| `topk` (0.01–1.0, default 0.2) | Fraction of key blocks each query block attends to exactly. 0.2 = top 20% of blocks. Lower is sparser and faster. 1.0 is (numerically) full attention. |
| `feature_map` | Feature map for the linear branch: `softmax`, `elu`, `relu`. |
| `block_size_q` | Query block size: 64 or 128. |
| `block_size_k` | Key block size. Fixed at 64 by the SLA kernel. |
| `enabled` | Turn the patch off to A/B test. |
| `start_percent` / `end_percent` (optional) | Sampling-progress window in which sparse attention is used. Steps outside it use regular attention. Raising `start_percent` (e.g. 0.2) keeps the early, layout-defining steps exact. |
| `min_tokens` (optional, default 1024) | Self-attention calls with fewer tokens stay on regular attention. Short sequences gain little from sparsity. |

### SageSLA Attention (Optimized)

Same inputs as SLA Attention, minus the block sizes: SageSLA picks them per GPU architecture (128×64, or 64×128 on Hopper).

### What gets sparse attention

The kernels are used only for **self-attention** calls with:

- head dimension 64 or 128 (Flux, Wan, HunyuanVideo, Qwen-Image, SD3 and SDXL all qualify; SD 1.5's 40/80/160 do not)
- no attention mask, no grouped-query attention, and the default softmax scale
- tensors on a CUDA device

Everything else (cross-attention to text, masked attention, other head sizes) runs on ComfyUI's normal attention backend. If another node has already installed an attention override (e.g. a different attention backend), fallback calls go to that override.

### Recommended Settings

| Goal | topk | start_percent |
|---|---|---|
| Quality first | 0.3 | 0.2 |
| Balanced | 0.2 | 0.1 |
| Maximum speed | 0.1 | 0.0 |

## 🐛 Troubleshooting

**"SLA module not found"**: install SLA:
```bash
pip install git+https://github.com/marduk191/SLA.git
```

**"SageSLA not available ... falling back to standard SLA"**: SpargeAttn isn't installed, or doesn't match your GPU or PyTorch. Standard SLA is used instead.

**"SLA failed (...); using regular attention for the rest of this model"**: the kernel raised an error, for example Triton is missing or there is no compiled kernel for your GPU. The node logs the error and keeps sampling on regular attention.

**No speedup**: check that the model runs on CUDA, that its head dim is 64/128, and that sequences are long (high resolution or video). Lower `topk`. Enable INFO logging to see `sparse attention on (B, H, L, D)` messages.

**Quality loss**: raise `topk`, raise `start_percent`, or try a different `feature_map`.

## 🧪 Testing

`tests/test_comfyui_integration.py` runs the nodes inside a ComfyUI checkout. It builds a small random-weight Flux model, samples with and without the patch, and checks both tensor layouts, the fallbacks, override chaining and the step window. Without a GPU it runs the real SLA Triton kernels through Triton's CPU interpreter:

```bash
COMFYUI_PATH=/path/to/ComfyUI python tests/test_comfyui_integration.py
```

On a CUDA machine, add `SLA_TEST_DEVICE=cuda` to run the kernels natively.

## 🙏 Credits

- **SLA**: [thu-ml/SLA](https://github.com/thu-ml/SLA). "SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention" (Zhang et al., 2025)
- **SpargeAttn / SageAttention**: [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn)
- **Windows wheels**: [woct0rdho/SpargeAttn](https://github.com/woct0rdho/SpargeAttn)

## 📄 License

Apache-2.0, following the [SLA repository](https://github.com/thu-ml/SLA).

## 🔗 Links

- [SLA Original Repository](https://github.com/thu-ml/SLA)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Comfy Registry](https://registry.comfy.org)
