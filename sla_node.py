"""
ComfyUI Node for SLA (Sparse-Linear Attention)
Implements attention acceleration for diffusion models using SLA
"""

import torch
import folder_paths
import comfy.model_patcher
import comfy.samplers
from comfy.model_patcher import ModelPatcher

class SLAAttentionNode:
    """
    A ComfyUI node that applies SLA (Sparse-Linear Attention) to models.
    This accelerates diffusion model inference by combining sparse and linear attention.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "topk": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Sparsity ratio (0.2 = keep 80% of tokens)"
                }),
                "feature_map": (["softmax", "elu", "relu"], {
                    "default": "softmax",
                    "tooltip": "Kernel type for attention computation"
                }),
                "block_size_q": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Block size for query processing"
                }),
                "block_size_k": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Block size for key processing"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable SLA attention"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_sla"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = "Applies SLA (Sparse-Linear Attention) to accelerate model inference"

    def __init__(self):
        self.sla_module = None

    def apply_sla(self, model, topk, feature_map, block_size_q, block_size_k, enabled):
        """
        Apply SLA attention to the model

        Args:
            model: ComfyUI MODEL object
            topk: Sparsity ratio (0.2 = 20% sparsity)
            feature_map: Type of feature map ('softmax', 'elu', 'relu')
            block_size_q: Block size for query
            block_size_k: Block size for key
            enabled: Whether to enable SLA

        Returns:
            Modified MODEL with SLA attention applied
        """
        if not enabled:
            return (model,)

        try:
            from sparse_linear_attention import SparseLinearAttention
        except ImportError:
            print("WARNING: SLA module not found. Please install with: pip install git+https://github.com/thu-ml/SLA.git")
            print("Returning unmodified model.")
            return (model,)

        # Clone the model to avoid modifying the original
        cloned_model = model.clone()

        # Get the actual model from the patcher
        model_object = cloned_model.model

        # Patch the attention mechanism
        def sla_attention_patch(q, k, v, extra_options):
            """
            Custom attention function using SLA

            Args:
                q: Query tensor [batch, heads, seq_len, head_dim]
                k: Key tensor [batch, heads, seq_len, head_dim]
                v: Value tensor [batch, heads, seq_len, head_dim]
                extra_options: Additional options from ComfyUI

            Returns:
                Attention output tensor
            """
            batch_size, num_heads, seq_len, head_dim = q.shape

            # Initialize SLA module if not already done or if parameters changed
            sla_key = (head_dim, topk, feature_map, block_size_q, block_size_k)
            if not hasattr(self, 'sla_cache') or self.sla_cache.get('key') != sla_key:
                self.sla_cache = {
                    'key': sla_key,
                    'module': SparseLinearAttention(
                        head_dim=head_dim,
                        topk=topk,
                        feature_map=feature_map,
                        BLKQ=block_size_q,
                        BLKK=block_size_k,
                    ).to(q.device).to(q.dtype)
                }

            sla_attn = self.sla_cache['module']

            # Apply SLA attention
            # SLA expects input in format [batch, heads, seq_len, head_dim]
            output = sla_attn(q, k, v)

            return output

        # Set the attention patch on the model
        cloned_model.set_model_attn1_patch(sla_attention_patch)

        print(f"SLA Attention applied with topk={topk}, feature_map={feature_map}, block_size=({block_size_q}, {block_size_k})")

        return (cloned_model,)


class SLASageAttentionNode:
    """
    A ComfyUI node that applies SageSLA (optimized SLA) to models.
    SageSLA is an optimized implementation based on SageAttention framework.
    Requires: pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
    Note: SageSLA only supports fixed block sizes (BLKQ=128, BLKK=64)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "topk": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Sparsity ratio (0.2 = keep 80% of tokens)"
                }),
                "feature_map": (["softmax", "elu", "relu"], {
                    "default": "softmax",
                    "tooltip": "Kernel type for attention computation"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable SageSLA attention"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_sage_sla"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = "Applies SageSLA (optimized SLA) for faster inference. Requires SpargeAttn installation."

    def __init__(self):
        self.sage_sla_module = None

    def apply_sage_sla(self, model, topk, feature_map, enabled):
        """
        Apply SageSLA (optimized) attention to the model

        Args:
            model: ComfyUI MODEL object
            topk: Sparsity ratio
            feature_map: Type of feature map ('softmax', 'elu', 'relu')
            enabled: Whether to enable SageSLA

        Returns:
            Modified MODEL with SageSLA attention applied
        """
        if not enabled:
            return (model,)

        try:
            # Try to import SageSLA
            try:
                from SageSLA import SageSparseLinearAttention
                use_sage = True
            except ImportError:
                # Fallback to regular SLA if SageSLA not available
                from sparse_linear_attention import SparseLinearAttention
                use_sage = False
                print("INFO: Using standard SLA (SageSLA not found)")
                print("INFO: For optimized SageSLA, install: pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation")
        except ImportError:
            print("WARNING: SLA module not found. Please install with: pip install git+https://github.com/thu-ml/SLA.git")
            print("Returning unmodified model.")
            return (model,)

        # Clone the model
        cloned_model = model.clone()

        # Patch the attention mechanism
        def sage_sla_attention_patch(q, k, v, extra_options):
            """Custom attention function using SageSLA"""
            batch_size, num_heads, seq_len, head_dim = q.shape

            # Initialize SageSLA module
            sage_key = (head_dim, topk, feature_map, use_sage)
            if not hasattr(self, 'sage_cache') or self.sage_cache.get('key') != sage_key:
                if use_sage:
                    # SageSLA only supports fixed block sizes: BLKQ=128, BLKK=64
                    self.sage_cache = {
                        'key': sage_key,
                        'module': SageSparseLinearAttention(
                            head_dim=head_dim,
                            topk=topk,
                            feature_map=feature_map,
                            BLKQ=128,  # Fixed for SageSLA
                            BLKK=64,   # Fixed for SageSLA
                        ).to(q.device).to(q.dtype)
                    }
                else:
                    # Standard SLA fallback with configurable block sizes
                    self.sage_cache = {
                        'key': sage_key,
                        'module': SparseLinearAttention(
                            head_dim=head_dim,
                            topk=topk,
                            feature_map=feature_map,
                            BLKQ=128,
                            BLKK=64,
                        ).to(q.device).to(q.dtype)
                    }

            sage_attn = self.sage_cache['module']
            output = sage_attn(q, k, v)

            return output

        cloned_model.set_model_attn1_patch(sage_sla_attention_patch)

        impl_type = "SageSLA (optimized)" if use_sage else "Standard SLA"
        print(f"{impl_type} Attention applied with topk={topk}, feature_map={feature_map}, block_size=(128, 64)")

        return (cloned_model,)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SLAAttention": SLAAttentionNode,
    "SLASageAttention": SLASageAttentionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLAAttention": "SLA Attention",
    "SLASageAttention": "SageSLA Attention (Optimized)",
}
