"""
Integration tests for the SLA-sage nodes against a ComfyUI checkout.

    COMFYUI_PATH=/path/to/ComfyUI python tests/test_comfyui_integration.py

Needs torch, ComfyUI's requirements, triton and the SLA package. Without a
GPU it runs the real SLA Triton kernels through Triton's CPU interpreter
(TRITON_INTERPRET=1) and lifts the node's CUDA-only guard; on a CUDA machine
set SLA_TEST_DEVICE=cuda to run the kernels natively.
"""

import os
import sys

DEVICE = os.environ.get("SLA_TEST_DEVICE", "cpu")
if DEVICE == "cpu":
    os.environ.setdefault("TRITON_INTERPRET", "1")

COMFYUI_PATH = os.environ.get("COMFYUI_PATH")
if not COMFYUI_PATH:
    sys.exit("Set COMFYUI_PATH to a ComfyUI checkout.")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFYUI_PATH)
sys.path.insert(0, REPO)
sys.argv = [sys.argv[0]] + (["--cpu"] if DEVICE == "cpu" else [])

import torch  # noqa: E402

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.cli_args  # noqa: E402,F401
import comfy.model_patcher  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.supported_models  # noqa: E402
from comfy.ldm.modules.attention import optimized_attention  # noqa: E402

import sla_node  # noqa: E402

if DEVICE == "cpu":
    _orig_reason = sla_node.SLAPatch.dense_reason

    def _reason_ignoring_device(self, *args, **kwargs):
        reason = _orig_reason(self, *args, **kwargs)
        return None if reason == "not on CUDA" else reason

    sla_node.SLAPatch.dense_reason = _reason_ignoring_device

    # Triton's CPU interpreter mishandles bf16 tensors, so compute in fp16 here.
    import sparse_linear_attention

    class _FP16SparseLinearAttention(sparse_linear_attention.SparseLinearAttention):
        def __init__(self, *args, **kwargs):
            kwargs["use_bf16"] = False
            super().__init__(*args, **kwargs)

    sparse_linear_attention.SparseLinearAttention = _FP16SparseLinearAttention

SPARSE_CALLS = []
_orig_module_for = sla_node.SLAPatch.module_for


def _counting_module_for(self, head_dim, device):
    module = _orig_module_for(self, head_dim, device)

    def run(q, k, v):
        out = module(q, k, v)          # only successful kernel calls are counted
        SPARSE_CALLS.append(head_dim)
        return out
    return run


sla_node.SLAPatch.module_for = _counting_module_for

torch.manual_seed(0)
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def tiny_flux():
    unet_config = dict(
        image_model="flux", in_channels=16, out_channels=16, vec_in_dim=32, context_in_dim=32,
        hidden_size=256, mlp_ratio=2.0, num_heads=2, depth=1, depth_single_blocks=1,
        axes_dim=[16, 56, 56], theta=10000, patch_size=2, qkv_bias=True, guidance_embed=True,
        txt_ids_dims=[],
    )
    config = comfy.supported_models.Flux(unet_config)
    config.set_inference_dtype(torch.float32, None)
    model = config.get_model({}, device=DEVICE)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.diffusion_model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device(DEVICE),
                                            offload_device=torch.device("cpu"))


def sample(model, steps=2):
    latent = torch.zeros(1, 16, 32, 32)          # 16x16 patches + 64 text tokens = 320 tokens
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(1))
    cond = [[torch.randn(1, 64, 32, generator=torch.Generator().manual_seed(2)),
             {"pooled_output": torch.randn(1, 32, generator=torch.Generator().manual_seed(3)),
              "guidance": 3.5}]]
    return comfy.sample.sample(model, noise, steps, 1.0, "euler", "simple", cond, cond, latent,
                               denoise=1.0, seed=1, disable_pbar=True)


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_node_schema():
    sla = sla_node.SLAAttentionNode.INPUT_TYPES()
    sage = sla_node.SLASageAttentionNode.INPUT_TYPES()
    check("SLA node inputs", list(sla["required"]) ==
          ["model", "topk", "feature_map", "block_size_q", "block_size_k", "enabled"])
    check("SageSLA node inputs", list(sage["required"]) == ["model", "topk", "feature_map", "enabled"])
    check("optional inputs", list(sla["optional"]) == ["start_percent", "end_percent", "min_tokens"])


def test_override_layouts():
    """optimized_attention -> override -> real SLA kernel, both tensor layouts."""
    model = tiny_flux()
    (patched,) = sla_node.SLAAttentionNode().apply_sla(model, 1.0, "softmax", 64, 64, True, min_tokens=0)
    to = patched.model_options["transformer_options"]
    check("override installed", "optimized_attention_override" in to)
    check("original model untouched",
          "optimized_attention_override" not in model.model_options["transformer_options"])

    b, h, n, d = 1, 2, 200, 64
    q, k, v = (torch.randn(b, h, n, d, device=DEVICE) for _ in range(3))
    ref = optimized_attention(q, k, v, h, skip_reshape=True, skip_output_reshape=True)

    SPARSE_CALLS.clear()
    out = optimized_attention(q, k, v, h, skip_reshape=True, skip_output_reshape=True, transformer_options=to)
    check("BHLD layout: sparse path taken", SPARSE_CALLS == [d])
    check("BHLD layout: topk=1 matches full attention", rel_err(out, ref) < 2e-2, f"rel_err={rel_err(out, ref):.2e}")

    flat = [t.transpose(1, 2).reshape(b, n, h * d) for t in (q, k, v)]
    ref_flat = optimized_attention(*flat, h)
    SPARSE_CALLS.clear()
    out_flat = optimized_attention(*flat, h, transformer_options=to)
    check("BL(HD) layout: sparse path taken", SPARSE_CALLS == [d])
    check("BL(HD) layout: shape", out_flat.shape == ref_flat.shape, str(tuple(out_flat.shape)))
    check("BL(HD) layout: matches full attention", rel_err(out_flat, ref_flat) < 2e-2,
          f"rel_err={rel_err(out_flat, ref_flat):.2e}")

    # Calls the kernel cannot serve must fall through to regular attention.
    SPARSE_CALLS.clear()
    kc, vc = torch.randn(b, h, 77, d, device=DEVICE), torch.randn(b, h, 77, d, device=DEVICE)
    out = optimized_attention(q, kc, vc, h, skip_reshape=True, transformer_options=to)
    ref = optimized_attention(q, kc, vc, h, skip_reshape=True)
    check("cross-attention falls back", SPARSE_CALLS == [] and torch.allclose(out, ref))

    q80 = torch.randn(b, h, n, 80, device=DEVICE)
    out = optimized_attention(q80, q80, q80, h, skip_reshape=True, transformer_options=to)
    check("head_dim 80 falls back", SPARSE_CALLS == [] and
          torch.allclose(out, optimized_attention(q80, q80, q80, h, skip_reshape=True)))

    mask = torch.zeros(n, n, device=DEVICE)
    out = optimized_attention(q, k, v, h, skip_reshape=True, mask=mask, transformer_options=to)
    check("masked attention falls back", SPARSE_CALLS == [])


def test_override_chaining():
    seen = []

    def previous(func, *args, **kwargs):
        seen.append(True)
        return func(*args, **kwargs)

    model = tiny_flux()
    model.model_options["transformer_options"]["optimized_attention_override"] = previous
    (patched,) = sla_node.SLAAttentionNode().apply_sla(model, 0.5, "softmax", 64, 64, True, min_tokens=0)
    to = patched.model_options["transformer_options"]
    q = torch.randn(1, 2, 200, 64, device=DEVICE)
    kc = torch.randn(1, 2, 77, 64, device=DEVICE)
    optimized_attention(q, kc, kc, 2, skip_reshape=True, transformer_options=to)
    check("fallback goes to the previously installed override", seen == [True])


def test_sampling():
    """Full ComfyUI sampling run on a (random-weight) Flux model."""
    model = tiny_flux()
    base = sample(model)

    SPARSE_CALLS.clear()
    (full,) = sla_node.SLAAttentionNode().apply_sla(model, 1.0, "softmax", 64, 64, True, min_tokens=0)
    out_full = sample(full)
    check("sampling: sparse kernel used in every attention call", len(SPARSE_CALLS) == 2 * 2 * 1,
          f"calls={len(SPARSE_CALLS)}")   # 2 steps x (1 double + 1 single block), cfg=1
    check("sampling: topk=1.0 ~ baseline", rel_err(out_full, base) < 3e-2, f"rel_err={rel_err(out_full, base):.2e}")

    for name, node, args in [
        ("SLA topk=0.2 q128", sla_node.SLAAttentionNode().apply_sla, (0.2, "softmax", 128, 64, True)),
        ("SLA topk=0.3 elu", sla_node.SLAAttentionNode().apply_sla, (0.3, "elu", 64, 64, True)),
        ("SageSLA (SLA fallback) topk=0.2 relu", sla_node.SLASageAttentionNode().apply_sage_sla,
         (0.2, "relu", True)),
    ]:
        SPARSE_CALLS.clear()
        (patched,) = node(model, *args, min_tokens=0)
        out = sample(patched)
        check(f"sampling: {name}", torch.isfinite(out).all().item() and len(SPARSE_CALLS) > 0
              and out.shape == base.shape, f"rel_err_vs_dense={rel_err(out, base):.2e} calls={len(SPARSE_CALLS)}")

    SPARSE_CALLS.clear()
    (windowed,) = sla_node.SLAAttentionNode().apply_sla(model, 0.2, "softmax", 64, 64, True,
                                                         start_percent=0.0, end_percent=0.4, min_tokens=0)
    sample(windowed)
    check("start/end_percent limits the sparse steps", len(SPARSE_CALLS) == 2, f"calls={len(SPARSE_CALLS)}")

    SPARSE_CALLS.clear()
    (big_min,) = sla_node.SLAAttentionNode().apply_sla(model, 0.2, "softmax", 64, 64, True, min_tokens=4096)
    out = sample(big_min)
    check("min_tokens keeps short sequences dense", SPARSE_CALLS == [] and torch.allclose(out, base))

    (off,) = sla_node.SLAAttentionNode().apply_sla(model, 0.2, "softmax", 64, 64, False)
    check("enabled=False passes the model through", off is model)


if __name__ == "__main__":
    print(f"torch {torch.__version__}, device {DEVICE}, TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET')}")
    test_node_schema()
    test_override_layouts()
    test_override_chaining()
    test_sampling()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall checks passed")
    sys.exit(1 if FAILED else 0)
