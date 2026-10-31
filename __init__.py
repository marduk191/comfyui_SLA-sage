"""
ComfyUI SLA-sage custom nodes.
Sparse-Linear Attention (SLA / SageSLA) patches for accelerating diffusion transformers.
"""

import logging

from .sla_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "1.1.0"

logging.getLogger("SLA-sage").info("Loaded SLA-sage nodes: SLA Attention, SageSLA Attention")
