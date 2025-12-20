"""
ComfyUI SLA-sage Custom Nodes
Provides Sparse-Linear Attention nodes for accelerating diffusion models
"""

from .sla_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.0.0"
__author__ = "ComfyUI SLA-sage"
__description__ = "Sparse-Linear Attention nodes for ComfyUI"

# Optional: Print loading message
print("\033[92m[SLA-sage]\033[0m Loaded SLA (Sparse-Linear Attention) nodes")
print("\033[92m[SLA-sage]\033[0m Available nodes: SLA Attention, SageSLA Attention")
