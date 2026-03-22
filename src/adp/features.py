import logging
import pickle
from typing import Dict, Tuple, Any, List, Optional
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
        # n_jobs=1: multiprocessing coordination overhead dominates for the small
        # batch sizes we use, making n_jobs=-1 slower than single-threaded.
        self.knn = NearestNeighbors(n_neighbors=1, metric='manhattan', n_jobs=1)
        self.knn.fit(state_matrix)

        self._cache: Dict[Tuple, np.ndarray] = {}
        self.seen_count = 0
        self.unseen_count = 0

    def _flatten_state(self, resource_state: Tuple) -> np.ndarray:
        """Flattens the nested state tuple, keeping ONLY numerical values for KNN distance."""
        flat = []

        def extract_numbers(item):
            if isinstance(item, (tuple, list)):
                for sub_item in item:
                    extract_numbers(sub_item)
            elif isinstance(item, (int, float, bool)):
                flat.append(float(item))

        extract_numbers(resource_state)
        return np.array(flat, dtype=np.float64)

    def _extract_batch(self, resource_states: List[Optional[Tuple]]) -> np.ndarray:
        """
        Core batch extraction from a list of resource_state tuples.

        Separates exact graph hits, cache hits, and KNN misses in a single pass,
        then issues ONE knn.kneighbors() call for all misses.
        """
        n = len(resource_states)
        result = np.zeros((n, self.num_features), dtype=np.float64)

        miss_indices = []
        miss_flat = []

        for i, resource_state in enumerate(resource_states):
            if resource_state is None:
                continue

            # 1. Exact graph node hit (O(1))
            idx = self._state_to_idx.get(resource_state)
            if idx is not None:
                self.seen_count += 1
                result[i] = self._basis_matrix[idx]
                continue

            # 2. Flatten and check secondary cache
            flat = self._flatten_state(resource_state)
            cache_key = tuple(np.round(flat, decimals=4))
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.seen_count += 1
                result[i] = cached
                continue

            miss_indices.append((i, resource_state, cache_key))
            miss_flat.append(flat)

        # Single batch KNN call for all misses
        if miss_flat:
            self.unseen_count += len(miss_flat)
            _, indices = self.knn.kneighbors(np.array(miss_flat))
            for (i, resource_state, cache_key), nearest_idx in zip(miss_indices, indices[:, 0]):
                features = self._basis_matrix[nearest_idx]
                result[i] = features
                self._cache[cache_key] = features
                self._state_to_idx[resource_state] = nearest_idx

        return result

    def extract_features(self, state: Any) -> np.ndarray:
        """Extract feature vector for a single state object."""
        resource_state = getattr(state, "resource_state", None)
        return self._extract_batch([resource_state])[0]

    def extract_features_batch(self, states: list) -> np.ndarray:
        """
        Extract feature vectors for a list of state objects in one KNN call.
        Used by the TD learner to batch the entire trajectory at once.
        """
        resource_states = [getattr(s, "resource_state", None) for s in states]
        return self._extract_batch(resource_states)

    def extract_resource_states_batch(self, resource_states: list) -> np.ndarray:
        """
        Extract feature vectors for a list of resource_state tuples directly.
        Used by the ADPPolicy lookahead, which produces resource_states without
        creating full AirportState objects.
        """
        return self._extract_batch(resource_states)

    def print_stats(self):
        total = self.seen_count + self.unseen_count
        if total > 0:
            unseen_pct = (self.unseen_count / total) * 100
            print(
                f"  [Extractor] State hits (Exact/Cache): {self.seen_count}, KNN Lookups: {self.unseen_count} ({unseen_pct:.1f}% unique)")
            print(f"  [Extractor] Cache size: {len(self._cache)}")
