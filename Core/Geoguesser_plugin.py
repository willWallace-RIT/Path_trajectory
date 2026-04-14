import random
import numpy as np

class GeoGuessrPlugin:
    """
    Simulates world anchoring from scene understanding.
    Replace this with your real plugin.
    """

    def predict_world_anchor(self, frame):
        # TODO: replace with real inference
        x = random.uniform(0, 5000)
        y = random.uniform(0, 5000)
        confidence = random.uniform(0.4, 0.9)

        return {
            "world": np.array([x, y]),
            "confidence": confidence
        }
