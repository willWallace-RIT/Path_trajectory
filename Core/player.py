import cv2
import numpy as np
from config import PLAYER_COLOR_LOWER, PLAYER_COLOR_UPPER

class PlayerDetector:
    def detect(self, frame):
        mask = cv2.inRange(frame, PLAYER_COLOR_LOWER, PLAYER_COLOR_UPPER)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.array([0.0, 0.0])

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        return np.array([x + w / 2, y + h / 2])
