# -*- coding: utf-8 -*-
"""
Meta Module
===========

Hierarchical emergence and renormalization group (RG) analysis for
multi-agent belief systems.

This module implements the novel theoretical contribution: how individual
agent beliefs combine to form emergent group-level "meta-agents" through
a principled renormalization procedure.

Key Components:
---------------
1. emergence.py: Core hierarchical meta-agent formation
2. consensus.py: Consensus belief computation with gauge transport
3. spatial_emergence.py: Spatial clustering and block structure
4. hierarchical_evolution.py: Scale-to-scale dynamics (RG flow)
5. visualization.py: Advanced plotting for meta-agent analysis

Sociology/Psychology Interpretation:
------------------------------------
The RG procedure captures:
- How subgroups form natural opinion clusters
- How group consensus emerges from individual beliefs
- Scale hierarchy: individual → subgroup → faction → society
- Why some belief differences persist at all scales (phase transitions)

Mathematical Foundation:
------------------------
At each scale ζ, we:
1. Cluster agents by belief similarity (low KL divergence)
2. Compute cluster consensus (gauge-invariant Fréchet mean)
3. Define meta-agents representing cluster beliefs
4. Iterate to coarser scales

This is analogous to:
- Block spin renormalization in physics
- Hierarchical clustering in statistics
- Multi-level modeling in sociology
"""

from .emergence import (
    HierarchicalSystem,
    MetaAgentDescriptor,
    ScaleIndex,
)

from .consensus import (
    compute_cluster_consensus,
    ConsensusConfig,
)

__all__ = [
    'HierarchicalSystem',
    'MetaAgentDescriptor',
    'ScaleIndex',
    'compute_cluster_consensus',
    'ConsensusConfig',
]
