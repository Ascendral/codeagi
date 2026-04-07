"""
Connective Tissue Layer — the wiring between KlomboAGI's brain systems.

This package connects the existing reasoning, memory, curiosity, and causal
systems into a unified thinking process. Each system already works on its own.
This layer makes them talk to each other.

Three modules:
- transfer_cortex: Cross-domain pattern recognition and prediction
- investigation: Deep recursive inquiry that follows knowledge threads
- nexus: The central hub that wires everything together

No LLM. Pure algorithmic integration of existing subsystems.
"""

from klomboagi.connective.transfer_cortex import TransferCortex, TransferablePattern, TransferPrediction
from klomboagi.connective.investigation import InvestigationEngine, InvestigationThread, InvestigationFinding
from klomboagi.connective.nexus import CognitiveNexus, AwarenessMemo
from klomboagi.connective.runtime_bridge import ConnectedRuntime, ConnectedAgent

__all__ = [
    "TransferCortex",
    "TransferablePattern",
    "TransferPrediction",
    "InvestigationEngine",
    "InvestigationThread",
    "InvestigationFinding",
    "CognitiveNexus",
    "AwarenessMemo",
    "ConnectedRuntime",
    "ConnectedAgent",
]
