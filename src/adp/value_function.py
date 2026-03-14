import numpy as np

class LinearVFA:
    """
    Linear Value Function Approximator.
    Maintains a weight vector to approximate values linearly from features.
    """

    def __init__(self, num_features: int) -> None:
        """
        Initialize the Linear VFA with a zero weight vector.

        Args:
            num_features (int): Number of features in the feature representation.
        """
        self.num_features: int = num_features
        self.theta: np.ndarray = np.zeros(num_features, dtype=np.float64)

    def predict(self, phi: np.ndarray) -> float:
        """
        Predict the value for a given feature vector.

        Args:
            phi (np.ndarray): The feature vector representation of a state.

        Returns:
            float: The predicted value (dot product of weights and features).
        """
        return float(np.dot(self.theta, phi))

    def update(self, phi: np.ndarray, target: float, alpha: float) -> None:
        """
        Perform the standard semi-gradient TD update.

        Args:
            phi (np.ndarray): The feature vector of the updated state.
            target (float): The TD target value.
            alpha (float): The learning rate.
        """
        prediction: float = self.predict(phi)
        error: float = target - prediction
        self.theta += alpha * error * phi
