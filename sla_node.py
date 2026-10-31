"""
ComfyUI nodes for SLA (Sparse-Linear Attention) and SageSLA.

The nodes install an ``optimized_attention_override`` on the model's
transformer_options. ComfyUI routes every call to ``optimized_attention``
(SD/SDXL UNets, Flux, Wan, HunyuanVideo, Qwen-Image, ... ) through that hook,
so the patch works across model families. Calls the SLA kernels cannot handle
(cross-attention, masks, GQA, unsupported head dims, CPU tensors, short
sequences, steps outside the start/end window) fall through to whatever
attention ComfyUI would have used otherwise.
"""

import logging
import math
import sys

import torch

logger = logging.getLogger("SLA-sage")

SLA_BLOCK_K = 64             # the SLA Triton kernel only supports BLKK=64
SLA_BLOCK_Q_CHOICES = [64, 128]
SUPPORTED_HEAD_DIMS = (64, 128)
FEATURE_MAPS = ["softmax", "elu", "relu"]


def _setup_spargeattn_compat():
    """
    Compatibility shim for SpargeAttn pre-built Windows wheels.

    Pre-built wheels from woct0rdho/SpargeAttn use architecture-specific modules
    (_qattn_sm80, _qattn_sm89, _qattn_sm90) instead of the generic _qattn module
    that source builds create. This shim detects the GPU architecture and
    registers the appropriate module as spas_sage_attn._qattn so upstream
    SageSLA code works without modification.

    Works on:
    - Linux/Windows source builds (uses _qattn directly)
    - Windows pre-built wheels (falls back to _qattn_smXX)
    - RTX 50xx Blackwell (sm120) via sm90 PTX compatibility
    - RTX 40xx Ada (sm89)
    - RTX 30xx Ampere (sm80/86)
    """
    try:
        import spas_sage_attn._qattn  # noqa: F401  (source build)
        return True
    except ImportError:
        pass

    if not torch.cuda.is_available():
        return False

    try:
        props = torch.cuda.get_device_properties(0)
        sm = props.major * 10 + props.minor
    except Exception:
        return False

    try:
        if sm >= 90:
            # Hopper (sm90) and Blackwell (sm120) - use sm90 via PTX compatibility
            import spas_sage_attn._qattn_sm90 as qattn
        elif sm >= 89:
            import spas_sage_attn._qattn_sm89 as qattn
        else:
            import spas_sage_attn._qattn_sm80 as qattn
        sys.modules['spas_sage_attn._qattn'] = qattn
        return True
    except ImportError:
        return False


def _import_sla():
    try:
        from sparse_linear_attention import SparseLinearAttention
        return SparseLinearAttention
    except ImportError as e:
        logger.warning(
            "SLA module not found (%s). Install it with: "
            "pip install git+https://github.com/marduk191/SLA.git", e)
        return None


def _import_sage_sla():
    _setup_spargeattn_compat()
    try:
        from SageSLA import SageSparseLinearAttention
        return SageSparseLinearAttention
    except ImportError as e:
        logger.info("SageSLA not available (%s); falling back to standard SLA. "
                    "Install SpargeAttn for the optimized kernels.", e)
        return None


class SLAPatch:
    """Settings, module cache and log state for one patched model."""

    def __init__(self, name, make_module, topk, block_q, block_k,
                 sigma_start, sigma_end, min_tokens):
        self.name = name
        self.make_module = make_module   # (head_dim, device) -> nn.Module
        self.topk = topk
        self.block_q = block_q
        self.block_k = block_k
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.min_tokens = min_tokens
        self.modules = {}
        self.disabled_reason = None
        self._logged = set()

    def log_once(self, key, level, message, *args):
        if key not in self._logged:
            self._logged.add(key)
            logger.log(level, "%s: " + message, self.name, *args)

    def module_for(self, head_dim, device):
        key = (head_dim, str(device))
        module = self.modules.get(key)
        if module is None:
            module = self.make_module(head_dim)
            # Keep proj_l in the kernels' compute dtype (bf16 by default) so it
            # does not depend on CUDA autocast to match the activations.
            module = module.to(device=device, dtype=getattr(module, "dtype", None)).eval()
            self.modules[key] = module
        return module

    def dense_reason(self, q, k, v, dim_head, tokens, mask, kwargs):
        """Why this call has to use the regular attention, or None."""
        if self.disabled_reason is not None:
            return self.disabled_reason
        if mask is not None:
            return "attention mask"
        if kwargs.get("enable_gqa", False):
            return "grouped-query attention"
        if q.shape != k.shape or q.shape != v.shape:
            return "cross-attention"
        if dim_head not in SUPPORTED_HEAD_DIMS:
            return f"head_dim {dim_head} not in {SUPPORTED_HEAD_DIMS}"
        if tokens < self.min_tokens:
            return f"{tokens} tokens < min_tokens {self.min_tokens}"
        if int(self.topk * math.ceil(tokens / self.block_k)) < 1:
            return f"{tokens} tokens too short for topk={self.topk}"
        scale = kwargs.get("scale", None)
        if scale is not None and not math.isclose(scale, dim_head ** -0.5, rel_tol=1e-4):
            return f"custom softmax scale {scale}"
        sigmas = (kwargs.get("transformer_options") or {}).get("sigmas", None)
        if sigmas is not None and len(sigmas) > 0:
            sigma = float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return "step outside start/end percent"
        if q.device.type != "cuda":
            return "not on CUDA"
        return None


def make_attention_override(patch, previous):
    """Build the optimized_attention_override. Calls SLA cannot serve go to the
    override that was installed before this one, or to ComfyUI's default."""

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
            if previous is not None:
                return previous(func, q, k, v, heads, **kw)
            return func(q, k, v, heads, **kw)

        if skip_reshape:                      # (B, H, L, D)
            b, n_heads, tokens, dim_head = q.shape
        else:                                 # (B, L, H*D)
            b, tokens, inner = q.shape
            n_heads = heads
            dim_head = inner // heads

        reason = patch.dense_reason(q, k, v, dim_head, tokens, mask, kwargs)
        if reason is not None:
            patch.log_once(("dense", reason), logging.DEBUG, "using regular attention (%s)", reason)
            return dense()

        if skip_reshape:
            qh, kh, vh = q, k, v
        else:
            qh, kh, vh = (t.view(b, -1, n_heads, dim_head).transpose(1, 2) for t in (q, k, v))

        try:
            out = patch.module_for(dim_head, q.device)(qh, kh, vh)
        except Exception as e:
            patch.disabled_reason = f"kernel error: {e}"
            logger.warning("%s failed (%s: %s); using regular attention for the rest of this model.",
                           patch.name, type(e).__name__, e)
            return dense()

        patch.log_once(("sparse", tuple(qh.shape)), logging.INFO,
                       "sparse attention on %s (B, H, L, D)", tuple(qh.shape))
        out = out.to(q.dtype)
        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(b, -1, n_heads * dim_head)

    return override


def apply_sla_patch(model, name, make_module, topk, block_q, block_k,
                    start_percent, end_percent, min_tokens):
    model_sampling = model.get_model_object("model_sampling")
    patch = SLAPatch(
        name, make_module, topk, block_q, block_k,
        sigma_start=float(model_sampling.percent_to_sigma(start_percent)),
        sigma_end=float(model_sampling.percent_to_sigma(end_percent)),
        min_tokens=min_tokens,
    )
    m = model.clone()
    to = m.model_options.setdefault("transformer_options", {})
    to["optimized_attention_override"] = make_attention_override(
        patch, to.get("optimized_attention_override"))
    return m


def _common_inputs(name):
    return {
        "topk": ("FLOAT", {
            "default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01,
            "tooltip": "Fraction of key blocks each query block attends to exactly "
                       "(0.2 = top 20% of blocks). Lower is faster and sparser."
        }),
        "feature_map": (FEATURE_MAPS, {
            "default": "softmax",
            "tooltip": "Feature map for the linear-attention branch."
        }),
    }, {
        "enabled": ("BOOLEAN", {
            "default": True,
            "tooltip": f"Enable/disable {name}. When disabled the model passes through unchanged."
        }),
    }


OPTIONAL_INPUTS = {
    "start_percent": ("FLOAT", {
        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": "Sampling progress at which sparse attention starts. "
                   "Raising it keeps the early, layout-defining steps on full attention."
    }),
    "end_percent": ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": "Sampling progress at which sparse attention stops."
    }),
    "min_tokens": ("INT", {
        "default": 1024, "min": 0, "max": 1 << 20, "step": 64,
        "tooltip": "Self-attention calls with fewer tokens than this use regular attention."
    }),
}


class SLAAttentionNode:
    """Patch a model to use SLA (Sparse-Linear Attention) Triton kernels."""

    @classmethod
    def INPUT_TYPES(cls):
        head, tail = _common_inputs("SLA attention")
        return {
            "required": {
                "model": ("MODEL",),
                **head,
                "block_size_q": (SLA_BLOCK_Q_CHOICES, {
                    "default": 64,
                    "tooltip": "Query block size for the sparse kernel (64 or 128)."
                }),
                "block_size_k": ([SLA_BLOCK_K], {
                    "default": SLA_BLOCK_K,
                    "tooltip": "Key block size. The SLA kernel only supports 64."
                }),
                **tail,
            },
            "optional": dict(OPTIONAL_INPUTS),
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_sla"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = ("Applies SLA (Sparse-Linear Attention) to the model's self-attention. "
                   "Requires a CUDA GPU and the SLA package. Unsupported calls use regular attention.")

    def apply_sla(self, model, topk, feature_map, block_size_q, block_size_k, enabled,
                  start_percent=0.0, end_percent=1.0, min_tokens=1024):
        if not enabled:
            return (model,)
        SparseLinearAttention = _import_sla()
        if SparseLinearAttention is None:
            logger.warning("SLA Attention: returning the unmodified model.")
            return (model,)

        block_size_q = int(block_size_q)
        block_size_k = int(block_size_k)

        def make_module(head_dim):
            return SparseLinearAttention(head_dim=head_dim, topk=topk, feature_map=feature_map,
                                         BLKQ=block_size_q, BLKK=block_size_k)

        logger.info("SLA attention: topk=%s feature_map=%s block_size=(%d, %d)",
                    topk, feature_map, block_size_q, block_size_k)
        return (apply_sla_patch(model, "SLA", make_module, topk, block_size_q, block_size_k,
                                start_percent, end_percent, min_tokens),)


class SLASageAttentionNode:
    """Patch a model to use SageSLA (SpargeAttn-based SLA), falling back to SLA."""

    @classmethod
    def INPUT_TYPES(cls):
        head, tail = _common_inputs("SageSLA attention")
        return {
            "required": {"model": ("MODEL",), **head, **tail},
            "optional": dict(OPTIONAL_INPUTS),
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_sage_sla"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = ("Applies SageSLA (quantized SLA kernels from SpargeAttn). Falls back to standard "
                   "SLA when SpargeAttn is not installed. Requires a CUDA GPU.")

    def apply_sage_sla(self, model, topk, feature_map, enabled,
                       start_percent=0.0, end_percent=1.0, min_tokens=1024):
        if not enabled:
            return (model,)

        SageSparseLinearAttention = _import_sage_sla()
        if SageSparseLinearAttention is not None:
            def make_module(head_dim):
                # SageSLA picks its own block sizes per GPU architecture.
                return SageSparseLinearAttention(head_dim=head_dim, topk=topk, feature_map=feature_map)
            name = "SageSLA"
            block_q, block_k = 128, 64
        else:
            SparseLinearAttention = _import_sla()
            if SparseLinearAttention is None:
                logger.warning("SageSLA Attention: neither SageSLA nor SLA is installed; "
                               "returning the unmodified model.")
                return (model,)

            def make_module(head_dim):
                return SparseLinearAttention(head_dim=head_dim, topk=topk, feature_map=feature_map,
                                             BLKQ=128, BLKK=SLA_BLOCK_K)
            name = "SLA (SageSLA fallback)"
            block_q, block_k = 128, SLA_BLOCK_K

        logger.info("%s attention: topk=%s feature_map=%s", name, topk, feature_map)
        return (apply_sla_patch(model, name, make_module, topk, block_q, block_k,
                                start_percent, end_percent, min_tokens),)


NODE_CLASS_MAPPINGS = {
    "SLAAttention": SLAAttentionNode,
    "SLASageAttention": SLASageAttentionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLAAttention": "SLA Attention",
    "SLASageAttention": "SageSLA Attention (Optimized)",
}
