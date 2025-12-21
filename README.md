# ComfyUI SLA-sage

**Sparse-Linear Attention (SLA) nodes for ComfyUI** - Accelerate your diffusion models with trainable sparse-linear attention!

This custom node package brings the power of [SLA (Sparse-Linear Attention)](https://github.com/thu-ml/SLA) to ComfyUI, enabling faster inference for Stable Diffusion and other diffusion transformer models.

## 🌟 Features

- **SLA Attention Node**: Full-featured sparse-linear attention with configurable parameters
- **SageSLA Attention Node**: Optimized version for maximum speed
- **Easy Integration**: Drop-in replacement for standard attention mechanisms
- **GPU Accelerated**: CUDA-optimized for best performance
- **ComfyUI Native**: Designed specifically for ComfyUI workflows

## 📋 What is SLA?

SLA (Sparse-Linear Attention) is a novel attention mechanism that combines sparse and linear attention to accelerate diffusion transformer models. It selectively attends to the most relevant tokens rather than computing full attention across all positions, resulting in:

- ⚡ **Faster inference** - Reduced computational complexity
- 🎯 **Maintained quality** - Minimal impact on generation quality
- 🔧 **Configurable** - Adjust sparsity levels to balance speed vs quality
- 🚀 **Trainable** - Can be fine-tuned for specific models

## 🔧 Installation

### Method 1: ComfyUI Manager (Recommended)

nope

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
   python install.py
   ```

   Or manually:
   ```bash
   pip install -r requirements.txt
   ```

4. Restart ComfyUI

### Requirements

- PyTorch >= 2.0.0
- CUDA-compatible GPU
- ComfyUI (latest version recommended)

### Optional: SageSLA (Optimized) Installation

For the best performance with the SageSLA node, you can install the optimized SpargeAttn backend:

```bash
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
```

**Note:**
- Without SpargeAttn, the SageSLA node will automatically fall back to using standard SLA
- The standard SLA implementation works well for most use cases
- SageSLA provides additional speed optimizations based on the SageAttention framework
- Installation requires a CUDA development environment

## 📖 Usage

### Available Nodes

#### 1. SLA Attention

The main SLA node with full configurability:

**Inputs:**
- `model` (MODEL): Input model to patch

**Parameters:**
- `topk` (float, 0.0-1.0): Sparsity ratio
  - 0.2 = keep 80% of tokens (recommended starting point)
  - Lower values = more sparse = faster but may reduce quality
  - Higher values = less sparse = slower but better quality

- `feature_map` (dropdown): Kernel type for attention
  - `softmax`: Standard attention-like kernel (recommended)
  - `elu`: ELU-based kernel
  - `relu`: ReLU-based kernel

- `block_size_q` (int, 16-256): Block size for query processing
  - Default: 64
  - Affects memory usage and performance

- `block_size_k` (int, 16-256): Block size for key processing
  - Default: 64
  - Affects memory usage and performance

- `enabled` (boolean): Enable/disable SLA
  - Useful for A/B testing

**Outputs:**
- `MODEL`: Patched model with SLA attention

#### 2. SageSLA Attention (Optimized)

Optimized implementation using fixed block sizes (BLKQ=128, BLKK=64) for best performance:

**Inputs:**
- `model` (MODEL): Input model to patch

**Parameters:**
- `topk` (float, 0.0-1.0): Sparsity ratio
- `feature_map` (dropdown): Kernel type (softmax, elu, relu)
- `enabled` (boolean): Enable/disable SageSLA

**Outputs:**
- `MODEL`: Patched model with SageSLA attention

**Note:** Automatically falls back to standard SLA if SpargeAttn is not installed. For optimal performance, install SpargeAttn (see Optional Installation above).

### Example Workflow

A basic workflow to use SLA:

```
CheckpointLoader → SLA Attention → KSampler → VAEDecode → SaveImage
```

1. Load your model with `CheckpointLoaderSimple`
2. Connect the MODEL output to `SLA Attention` node
3. Connect the SLA Attention output to your `KSampler`
4. Continue your workflow as normal

See `example_workflow.json` for a complete example you can drag into ComfyUI.

### Recommended Settings

**For best quality with good speedup:**
- topk: 0.2
- feature_map: softmax
- block sizes: 64 (default)

**For maximum speed:**
- topk: 0.1
- feature_map: relu
- block sizes: 32

**For minimal quality loss:**
- topk: 0.3
- feature_map: softmax
- block sizes: 64

## 🎯 Tips & Best Practices

1. **Start with defaults**: The default settings (topk=0.2, softmax) work well for most models

2. **Experiment with topk**: This is the most impactful parameter
   - Lower = faster but may affect quality
   - Higher = slower but better quality

3. **Use SageSLA for simplicity**: If you don't need fine control, SageSLA provides good defaults

4. **Test before production**: Always compare outputs with and without SLA to ensure quality meets your needs

5. **GPU Memory**: SLA may use different memory patterns than standard attention. If you encounter OOM errors, try:
   - Reducing block sizes
   - Increasing topk (less sparse)
   - Reducing batch size

## 🐛 Troubleshooting

### "SLA module not found" error

Make sure you've installed the SLA package:
```bash
pip install git+https://github.com/thu-ml/SLA.git
```

### "Using standard SLA (SageSLA not found)" message

This is an informational message, not an error. It means:
- You're using the SageSLA node without the SpargeAttn package installed
- The node automatically falls back to standard SLA implementation
- Everything will work fine, but you may not get the maximum performance boost

To get the optimized SageSLA implementation:
```bash
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
```

### No speedup observed

- Ensure you're using a CUDA GPU
- Try lower topk values (0.1-0.2)
- Some models may benefit more than others

### Quality degradation

- Increase topk value (0.3-0.4)
- Try different feature_map options
- Use SLA Attention with softmax kernel

### Out of memory errors

- Reduce block_size_q and block_size_k
- Increase topk (less sparse requires less memory overhead)
- Reduce batch size in your workflow

## 📚 Technical Details

### How It Works

SLA replaces the standard attention mechanism in diffusion models with a hybrid sparse-linear attention:

1. **Sparse Selection**: Identifies most relevant tokens using topk selection
2. **Linear Attention**: Applies efficient linear attention kernels
3. **Hybrid Output**: Combines sparse and linear components

This approach maintains quality while significantly reducing computational cost, especially for long sequences.

### Performance Characteristics

- **Speedup**: Typically 1.5-3x faster inference depending on settings
- **Memory**: Similar or slightly lower memory usage compared to standard attention
- **Quality**: Minimal quality loss with proper parameter tuning (topk ≥ 0.2)

## 🙏 Credits

- **SLA Implementation**: [thu-ml/SLA](https://github.com/thu-ml/SLA)
- **Original Paper**: "SLA: Sparse-Linear Attention for Accelerating Diffusion Models" (2025)
- **Authors**: Tsinghua University & UC Berkeley researchers

## 📄 License

This project follows the license of the original SLA repository. Please refer to the [SLA repository](https://github.com/thu-ml/SLA) for license details.

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs
- Suggest features
- Submit pull requests
- Share your results

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [example_workflow.json](example_workflow.json)
3. Open an issue on GitHub with:
   - ComfyUI version
   - GPU model
   - Error messages
   - Your workflow/settings

## 🔗 Links

- [SLA Original Repository](https://github.com/thu-ml/SLA)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

---

**Made with ❤️ for the ComfyUI community**

*Accelerate your diffusion models with SLA!*
