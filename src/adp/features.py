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
        resource_state: Tuple = getattr(state, "resource_state", None)
        if resource_state is None:
            return np.zeros(self.num_features, dtype=np.float64)

        # 1. Try exact match from the original graph (O(1))
        # We still want to do this first because it bypasses flattening entirely if we get a hit
        idx = self._state_to_idx.get(resource_state)
        if idx is not None:
            self.seen_count += 1
            return self._basis_matrix[idx]

        # 2. State is not an exact graph node. Flatten it for KNN and check cache.
        flat_state = self._flatten_state(resource_state)
        
        # Create a hashable cache key from the numerical array
        # This strips out 't' and specific string IDs that ruin cache hit rates
        cache_key = tuple(np.round(flat_state, decimals=4).flatten())
        
        if cache_key in self._cache:
            self.seen_count += 1 # A cache hit effectively counts as "seen" instantly
            return self._cache[cache_key]

        # 3. Cache Miss: Run KNN (Expensive!)
        self.unseen_count += 1
        flat_state_2d = flat_state.reshape(1, -1)
        _, indices = self.knn.kneighbors(flat_state_2d)
        nearest_idx = indices[0][0]
        
        features = self._basis_matrix[nearest_idx]

        # 4. Store result in cache before returning
        self._cache[cache_key] = features
        
        # Also store the exact mapping so step 1 catches it next time (optimization upon optimization!)
        self._state_to_idx[resource_state] = nearest_idx
        
        return features

    def print_stats(self):
        total = self.seen_count + self.unseen_count
        if total > 0:
            unseen_pct = (self.unseen_count / total) * 100
            print(
                f"  [Extractor] State hits (Exact/Cache): {self.seen_count}, KNN Lookups: {self.unseen_count} ({unseen_pct:.1f}% unique)")
            print(f"  [Extractor] Cache size: {len(self._cache)}")