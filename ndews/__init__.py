# ndews package — Neural Degeneration Early Warning System
"""
ndews — model-agnostic early warning for training instability and representational
collapse in PyTorch models.

Public API
----------
    from ndews import CollapseMonitor, MonitorReport, suggest_layers
"""

from __future__ import annotations

from ndews.monitor import CollapseMonitor, MonitorReport, suggest_layers

__version__ = "0.2.0"

__all__ = [
    "CollapseMonitor",
    "MonitorReport",
    "suggest_layers",
    "__version__",
]
