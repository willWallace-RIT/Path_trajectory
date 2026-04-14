import cv2
import json
from core.camera import CameraEstimator
from core.player import PlayerDetector
from core.fusion import WorldFusion
from core.geoguessr_plugin import GeoGuessrPlugin
from config import VIDEO_PATH, FRAME_SKIP

cap = cv2.VideoCapture(VIDEO_PATH)

camera = CameraEstimator()
player = PlayerDetector()
fusion = WorldFusion()
geo = GeoGuessrPlugin()

trajectory = []

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_SKIP != 0:
        frame_id += 1
        continue

    geo_data = geo.predict_world_anchor(frame)
    geo_anchor = geo_data["world"]

    cam = camera.estimate(frame)
    player_pos = player.detect(frame)

    world = fusion.update(geo_anchor, cam, player_pos)

    trajectory.append({
        "frame": frame_id,
        "world_x": float(world[0]),
        "world_y": float(world[1])
    })

    cv2.circle(frame, (int(player_pos[0]), int(player_pos[1])), 5, (0,255,0), -1)
    cv2.imshow("view", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

with open("output/trajectory.json", "w") as f:
    json.dump(trajectory, f, indent=2)
