"""
Aircraft-gate compatibility and preference configuration.

This module handles the filtering and validation of compatibility matrices,
ensuring that only the aircraft types active in the current scenario are 
loaded into the simulation's state. It provides both high-level string-based
lookups (for setup/logging) and optimized integer-based lookups (for the MDP inner loop).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import List, Union
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CompatibilityConfig:
    """
    Immutable configuration for aircraft-gate compatibility.

    Stores the subset of compatibility and preference data relevant to the 
    active fleet mix of the current scenario.

    Attributes:
        active_aircraft_types: List of aircraft type names present in this config.
        compatibility_matrix: Binary matrix (num_active_types x num_gates).
                              1 = Compatible, 0 = Incompatible.
        preference_matrix: Float matrix (num_active_types x num_gates).
                           Higher values indicate stronger preference.
    """
    
    active_aircraft_types: List[str]
    compatibility_matrix: npt.NDArray[np.int_]
    preference_matrix: npt.NDArray[np.float_]

    @cached_property
    def type_to_idx(self) -> dict[str, int]:
        """
        Mapping for O(1) row lookups by aircraft type name.
        
        Cached on first access to avoid computation during initialization.
        """
        return {name: idx for idx, name in enumerate(self.active_aircraft_types)}

    @classmethod
    def from_raw_data(
        cls,
        raw_comp_matrix: Union[List[List[int]], npt.NDArray],
        raw_pref_matrix: Union[List[List[float]], npt.NDArray],
        master_aircraft_list: List[str],
        accepted_aircraft_types: List[str]
    ) -> CompatibilityConfig:
        """
        Create a configuration by slicing raw matrices for specific aircraft types.

        This factory method takes the full "master" matrices (containing all known
        aircraft types) and filters them down to only include the rows corresponding
        to the `accepted_aircraft_types` for the current scenario.

        Args:
            raw_comp_matrix: Full compatibility matrix (rows=master_types).
            raw_pref_matrix: Full preference matrix (rows=master_types).
            master_aircraft_list: List of all aircraft types corresponding to raw rows.
            accepted_aircraft_types: List of types to include in this configuration.

        Returns:
            A new CompatibilityConfig instance with sliced matrices.

        Raises:
            ValueError: If dimensions mismatch or an accepted type is missing from master.
        """
        # Convert inputs to numpy arrays if they aren't already
        comp_arr = np.array(raw_comp_matrix, dtype=np.int_)
        pref_arr = np.array(raw_pref_matrix, dtype=np.float64)

        # Validate master dimensions
        if len(master_aircraft_list) != comp_arr.shape[0]:
            raise ValueError(
                f"Master aircraft list length ({len(master_aircraft_list)}) "
                f"does not match compatibility matrix rows ({comp_arr.shape[0]})."
            )
        
        # Validate that preference matrix matches compatibility matrix shape
        if comp_arr.shape != pref_arr.shape:
             raise ValueError(
                f"Raw matrix shape mismatch: Compatibility {comp_arr.shape} "
                f"vs Preference {pref_arr.shape}."
            )

        # Optimize lookup: Create a map for O(1) index retrieval
        master_type_map = {name: idx for idx, name in enumerate(master_aircraft_list)}

        # Build the row indices for slicing
        row_indices = []
        for aircraft_type in accepted_aircraft_types:
            try:
                idx = master_type_map[aircraft_type]
                row_indices.append(idx)
            except KeyError:
                raise ValueError(
                    f"Accepted aircraft type '{aircraft_type}' not found in "
                    f"master aircraft list: {master_aircraft_list}"
                )

        # Slice the matrices
        # We use advanced indexing: matrix[[row_indices], :]
        sliced_comp = comp_arr[row_indices, :]
        sliced_pref = pref_arr[row_indices, :]
        
        # Enforce True Immutability on the arrays
        sliced_comp.flags.writeable = False
        sliced_pref.flags.writeable = False

        return cls(
            active_aircraft_types=accepted_aircraft_types,
            compatibility_matrix=sliced_comp,
            preference_matrix=sliced_pref
        )

    # --- High-Level String APIs (Setup/Logging) ---

    def is_compatible(self, aircraft_type: str, gate_idx: int) -> bool:
        """
        Check if an aircraft type is physically compatible with a gate.

        Args:
            aircraft_type: Name of the aircraft type (e.g., 'B747').
            gate_idx: Index of the gate (column in the matrix).

        Returns:
            True if compatible, False otherwise.

        Raises:
            KeyError: If aircraft_type is not in the active configuration.
            IndexError: If gate_idx is out of bounds.
        """
        row_idx = self.type_to_idx[aircraft_type]
        return bool(self.compatibility_matrix[row_idx, gate_idx])

    def get_preference(self, aircraft_type: str, gate_idx: int) -> float:
        """
        Get the preference score for assigning an aircraft type to a gate.

        Args:
            aircraft_type: Name of the aircraft type.
            gate_idx: Index of the gate.

        Returns:
            Preference score (higher is better).

        Raises:
            KeyError: If aircraft_type is not in the active configuration.
            IndexError: If gate_idx is out of bounds.
        """
        row_idx = self.type_to_idx[aircraft_type]
        return float(self.preference_matrix[row_idx, gate_idx])

    def get_compatible_gates(self, aircraft_type: str) -> npt.NDArray[np.int_]:
        """
        Get all gate indices compatible with a specific aircraft type.

        Args:
            aircraft_type: Name of the aircraft type.

        Returns:
            Array of compatible gate indices.
        """
        row_idx = self.type_to_idx[aircraft_type]
        # Return indices where value is 1 (True)
        return np.where(self.compatibility_matrix[row_idx, :] == 1)[0]
        
    # --- Low-Level Integer APIs (Inner Loop Performance) ---

    def is_compatible_idx(self, ac_idx: int, gate_idx: int) -> bool:
        """
        Fast compatibility check using integer indices.
        
        Optimized for the MDP inner loop. Avoids string hashing and dictionary lookups.
        
        Args:
            ac_idx: Index of the aircraft type in the active configuration.
            gate_idx: Index of the gate.
            
        Returns:
            True if compatible.
        """
        return bool(self.compatibility_matrix[ac_idx, gate_idx])

    def get_preference_idx(self, ac_idx: int, gate_idx: int) -> float:
        """
        Fast preference lookup using integer indices.
        
        Optimized for the MDP inner loop.
        
        Args:
            ac_idx: Index of the aircraft type in the active configuration.
            gate_idx: Index of the gate.
            
        Returns:
            Preference score.
        """
        return float(self.preference_matrix[ac_idx, gate_idx])
