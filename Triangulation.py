 
import cv2
import numpy as np

def performTriangulation(camera_pose1, camera_pose2, keypoints1, keypoints2):
    reconstructed_points = np.zeros((keypoints1.shape[0], 4))
    camera_pose1_inv = np.linalg.inv(camera_pose1)
    camera_pose2_inv = np.linalg.inv(camera_pose2)

    for i, points in enumerate(zip(keypoints1, keypoints2)):
        triangulation_matrix = np.zeros((4, 4))
        triangulation_matrix[0] = points[0][0] * camera_pose1_inv[2] - camera_pose1_inv[0]
        triangulation_matrix[1] = points[0][1] * camera_pose1_inv[2] - camera_pose1_inv[1]
        triangulation_matrix[2] = points[1][0] * camera_pose2_inv[2] - camera_pose2_inv[0]
        triangulation_matrix[3] = points[1][1] * camera_pose2_inv[2] - camera_pose2_inv[1]

        _, _, vt = np.linalg.svd(triangulation_matrix)
        reconstructed_points[i] = vt[3]

    return reconstructed_points
