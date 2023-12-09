 
import numpy as np
import cv2

def extractFeatures(image):
    orb_detector = cv2.ORB_create()
    points = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=7)
    keypoints = [cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=20) for point in points]
    keypoints, orb_descriptors = orb_detector.compute(image, keypoints)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints]), orb_descriptors

def normalizePoints(inverse_camera_matrix, points):
    return np.dot(inverse_camera_matrix, addOnes(points).T).T[:, 0:2]

def denormalizePoints(camera_matrix, point):
    normalized_pt = np.dot(camera_matrix, np.array([point[0], point[1], 1.0]))
    normalized_pt /= normalized_pt[2]
    return int(round(normalized_pt[0])), int(round(normalized_pt[1]))

def addOnes(points):
    return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

identity_matrix = np.eye(4)

class CameraFrame:
    def __init__(self, frame_registry, image, camera_matrix):
        self.camera_matrix = camera_matrix
        self.inverse_camera_matrix = np.linalg.inv(self.camera_matrix)
        self.frame_pose = identity_matrix
        self.image_height, self.image_width = image.shape[0:2]
        keypoints, self.feature_descriptors = extractFeatures(image)
        self.normalized_keypoints = normalizePoints(self.inverse_camera_matrix, keypoints)
        self.tracked_points = [None] * len(self.normalized_keypoints)
        self.frame_id = len(frame_registry.frames)
        frame_registry.frames.append(self)
