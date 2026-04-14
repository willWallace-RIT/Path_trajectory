import cv2
import numpy as np

class CameraEstimator:
    def __init__(self):
        self.prev_gray = None

    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.array([0.0, 0.0, 0.0])

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        dx = float(np.mean(flow[..., 0]))
        dy = float(np.mean(flow[..., 1]))

        angle = float(np.arctan2(dy, dx + 1e-6))

        self.prev_gray = gray

        return np.array([dx, dy, angle])
