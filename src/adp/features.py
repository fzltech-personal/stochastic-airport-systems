import logging
import pickle
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class PVFFeatureExtractor:
    """
    Feature Extractor using Pre-computed Proto-Value Functions (PVFs).
    Uses K-Nearest Neighbors (KNN) to generalize features for unseen states.
    """

    def __init__(self, basis_matrix_path: str, state_mapping_path: str) -> None:
        self._basis_matrix: np.ndarray = np.load(basis_matrix_path)
        self.num_features: int = self._basis_matrix.shape[1]

        with open(state_mapping_path, "rb") as f:
            self.state_list: list = pickle.load(f)

        self._state_to_idx: Dict[Tuple, int] = {
            resource_state: idx for idx, resource_state in enumerate(self.state_list)
        }

        print("  [Extractor] Building KNN tree for unseen state generalization...")
        state_matrix = np.array([self._flatten_state(s) for s in self.state_list])
        self.knn = NearestNeighbors(n_neighbors=1, metric='manhattan', n_jobs=-1)
        self.knn.fit(state_matrix)

        # Cache for memoization
        self._cache: Dict[Tuple, np.ndarray] = {}
        self.seen_count = 0
        self.unseen_count = 0

    def _flatten_state(self, resource_state: Tuple) -> np.ndarray:
        """Flattens the nested state, keeping ONLY numerical values for KNN distance."""
        flat = []

        def extract_numbers(item):
            if isinstance(item, (tuple, list)):
                for sub_item in item:
                    extract_numbers(sub_item)
            elif isinstance(item, (int, float, bool)):
                flat.append(float(item))

        extract_numbers(resource_state)
        return np.array(flat, dtype=np.float64)

    def extract_features(self, state: Any) -> np.ndarray:
        """Extracts feature vector phi(s) for a given state, with caching."""
        # Use a hashable representation of the state as the cache key
        cache_key = (state.t, state.gates, state.runway_queue)
        if cache_key in self._cache:
            return self._cache[cache_key]

        resource_state: Tuple = getattr(state, "resource_state", None)
        if resource_state is None:
            return np.zeros(self.num_features, dtype=np.float64)

        idx = self._state_to_idx.get(resource_state)
        if idx is not None:
            self.seen_count += 1
            features = self._basis_matrix[idx]
        else:
            self.unseen_count += 1
            flat_state = self._flatten_state(resource_state).reshape(1, -1)
            _, indices = self.knn.kneighbors(flat_state)
            nearest_idx = indices[0][0]
            self._state_to_idx[resource_state] = nearest_idx
            features = self._basis_matrix[nearest_idx]

        # Store result in cache before returning
        self._cache[cache_key] = features
        return features

    def print_stats(self):
        total = self.seen_count + self.unseen_count
        if total > 0:
            unseen_pct = (self.unseen_count / total) * 100
            print(
                f"  [Extractor] State hits: Exact={self.seen_count}, KNN Generalization={self.unseen_count} ({unseen_pct:.1f}%)")
            print(f"  [Extractor] Cache size: {len(self._cache)}")