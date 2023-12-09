 
import numpy as np
import cv2
import os, sys, time, g2o
from triangulation import performTriangulation
from Camera import denormalizePoints, normalizePoints, Camera
from display import FrameDisplay
from match_frames import match_keypoints
from descriptor import WorldDescriptor, WorldPoint

F = int(os.getenv("F", "500"))
W, H = 1920 // 2, 1080 // 2
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
world_descriptor = WorldDescriptor()

world_descriptor.create_viewer()

display = FrameDisplay(W, H)

def preprocessImage(image):
    resized_image = cv2.resize(image, (W, H))
    return resized_image

def processSLAM(image):
    processed_image = preprocessImage(image)
    current_frame = Camera(world_descriptor, processed_image, K)
    if current_frame.camera_id == 0:
        return
    previous_frame = world_descriptor.frames[-2]

    matched_kp1, matched_kp2, relative_pose = match_keypoints(previous_frame, current_frame)
    current_frame.camera_pose = np.dot(relative_pose, previous_frame.camera_pose)

    for i, idx in enumerate(matched_kp1):  # Use matched_kp1 for previous_frame
        if previous_frame.tracked_points[idx] is not None:
            previous_frame.tracked_points[idx].add_observation(current_frame, matched_kp2[i])  # Use matched_kp2 for current_frame


    points_4d = performTriangulation(current_frame.camera_pose, previous_frame.camera_pose, previous_frame.normalized_keypoints[matched_kp1], current_frame.normalized_keypoints[matched_kp2])
    valid_4d_points = points_4d[:, 3] != 0
    points_4d = points_4d[valid_4d_points]
    matched_kp1 = matched_kp1[valid_4d_points]
    matched_kp2 = matched_kp2[valid_4d_points]
    points_4d /= points_4d[:, 3:]

    new_points = np.array([current_frame.tracked_points[i] is None if i < len(current_frame.tracked_points) else False for i in matched_kp1])

    # Before the loop where you use matched_kp1, add a check:
    if any(i >= len(current_frame.tracked_points) for i in matched_kp1):
        print("Index out of range error: matched_kp1 contains indices that are too large")
        return


    valid_points = (np.abs(points_4d[:, 3]) > 0.005) & (points_4d[:, 2] > 0) & new_points
    for i, point in enumerate(points_4d):
        if not valid_points[i]:
            continue
        new_point = WorldPoint(world_descriptor, point)
        # Ensure indices are in range
        if matched_kp1[i] < len(previous_frame.tracked_points) and matched_kp2[i] < len(current_frame.tracked_points):
            new_point.add_observation(previous_frame, matched_kp1[i])
            new_point.add_observation(current_frame, matched_kp2[i])
        else:
            print(f"Index out of range: {matched_kp1[i]} or {matched_kp2[i]}")


    for pt1, pt2 in zip(previous_frame.normalized_keypoints[matched_kp1], current_frame.normalized_keypoints[matched_kp2]):
        u1, v1 = denormalizePoints(K, pt1)
        u2, v2 = denormalizePoints(K, pt2)
        cv2.circle(processed_image, (u1, v1), color=(0, 255, 0), radius=1)
        cv2.line(processed_image, (u1, v1), (u2, v2), color=(255, 255, 0))

    if display is not None:
        display.show2D(processed_image)
    world_descriptor.update_display()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s requires a .mp4 file as an argument" % sys.argv[0])
        exit(-1)

    video_capture = cv2.VideoCapture(sys.argv[1])
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            cv2.imshow("Preview", cv2.resize(frame, (720, 400)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            processSLAM(frame)
        else:
            break
    video_capture.release()
    cv2.destroyAllWindows()
