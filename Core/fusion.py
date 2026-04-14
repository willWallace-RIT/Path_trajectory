import numpy as np
from config import WORLD_SMOOTHING

class WorldFusion:
    def __init__(self):
        self.world_pos = np.array([0.0, 0.0])

    def rotate(self, vec, angle):
        r = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        return r @ vec

    def update(self, geo_anchor, camera, player_screen):

        cam_motion = camera[:2]
        cam_angle = camera[2]

        # 1. Geo anchor correction (absolute grounding)
        self.world_pos = (
            (1 - WORLD_SMOOTHING) * self.world_pos +
            WORLD_SMOOTHING * geo_anchor
        )

        # 2. Camera motion contribution
        self.world_pos += cam_motion

        # 3. Player offset transformed into world space
        player_world_delta = self.rotate(player_screen, cam_angle)
        self.world_pos += player_world_delta * 0.01  # scale factor

        return self.world_pos.copy()
