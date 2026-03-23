"""
Graph construction utilities for Proto-Value Functions (PVFs).

This module handles the construction of a state-transition graph from
simulation trajectories. It collapses the time dimension to focus on
recurrent resource states (gate occupancy, runway queue).
Optimized for memory efficiency using sparse matrices.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Set
import networkx as nx
import numpy as np
import scipy.sparse
import numpy.typing as npt

from src.mdp.state import AirportState
from src.mdp.action import Action


class StateGraph:
    """
    Constructs a weighted undirected graph from MDP trajectories.

    Nodes represent unique resource states (gates, queue) independent of time.
    Edges represent observed transitions between these states.
    Weights correspond to the frequency of transitions.

    When a coarsener is provided, raw resource_states are mapped to compact
    coarsened tuples before being added to the graph.  This dramatically
    increases node revisitation frequency and produces a well-connected graph
    with meaningful PVF eigenvectors.  When coarsener=None the original
    behaviour is preserved (no regression for callers that omit it).
    """

    def __init__(self, coarsener=None) -> None:
        """
        Args:
            coarsener: Optional CoarsenedStateBuilder.  If supplied, every
                       resource_state is passed through coarsener.coarsen()
                       before being used as a graph node identity.
        """
        self.graph = nx.Graph()
        self._coarsener = coarsener

    def add_trajectory(self, trajectory: List[Tuple[AirportState, Action, float, AirportState]]):
        """
        Incorporate a new trajectory into the graph.

        Iterates through the transitions (s, a, r, s') and adds edges between
        the (optionally coarsened) resource states of s and s'.

        Args:
            trajectory: A list of (state, action, reward, next_state) tuples.
        """
        for state, action, reward, next_state in trajectory:
            # Skip terminal transitions (next_state is None for last step)
            if next_state is None:
                continue
            if self._coarsener is not None:
                u = self._coarsener.coarsen(state.resource_state)
                v = self._coarsener.coarsen(next_state.resource_state)
            else:
                # Legacy behaviour: raw resource_state as node identity
                u = state.resource_state
                v = next_state.resource_state

            # Add edge or increment weight
            if self.graph.has_edge(u, v):
                self.graph[u][v]['weight'] += 1
            else:
                self.graph.add_edge(u, v, weight=1)

    def get_adjacency_matrix(self) -> Tuple[scipy.sparse.spmatrix, List[Tuple]]:
        """
        Convert the graph to a sparse adjacency matrix.
        
        Uses SciPy sparse matrices (CSR format) to avoid OOM errors on large
        state spaces.

        Returns:
            adjacency_matrix: A sparse square matrix of shape (N, N) where
                              A[i, j] is the weight of the edge between node i and j.
            nodes: A list of length N containing the resource state tuples
                   corresponding to the rows/columns of the matrix.
        """
        # Get nodes in a deterministic order
        nodes = list(self.graph.nodes())
        
        # Create sparse adjacency matrix (CSR format is efficient for arithmetic)
        # weight='weight' ensures we capture transition frequencies
        adj_matrix = nx.to_scipy_sparse_array(
            self.graph, 
            nodelist=nodes, 
            weight='weight', 
            format='csr'
        )
        
        return adj_matrix, nodes

    @property
    def num_nodes(self) -> int:
        """Return the number of unique resource states discovered."""
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Return the number of unique transitions observed."""
        return self.graph.number_of_edges()
