 
import cv2
import numpy as np
np.set_printoptions(suppress=True)

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

def augment_with_ones(matrix):
    return np.concatenate([matrix, np.ones((matrix.shape[0], 1))], axis=1)

def decomposeFundamentalMatrix(F):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, _, Vt = np.linalg.svd(F)
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    return transformation

def match_keypoints(frame1, frame2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(frame1.feature_descriptors, frame2.feature_descriptors, k=2)

    good_matches = []
    keypoints1, keypoints2 = [], []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pt1 = frame1.normalized_keypoints[m.queryIdx]
            pt2 = frame2.normalized_keypoints[m.trainIdx]

            if np.linalg.norm((pt1 - pt2)) < 0.1 * np.linalg.norm([frame1.image_width, frame1.image_height]) and m.distance < 32:
                if m.queryIdx not in keypoints1 and m.trainIdx not in keypoints2:
                    keypoints1.append(m.queryIdx)
                    keypoints2.append(m.trainIdx)
                    good_matches.append((pt1, pt2))

    assert(len(set(keypoints1)) == len(keypoints1))
    assert(len(set(keypoints2)) == len(keypoints2))
    assert len(good_matches) >= 8

    good_matches = np.array(good_matches)
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)

    model, inliers = ransac((good_matches[:, 0], good_matches[:, 1]),
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,
                            max_trials=100)

    print("Matches: %d -> %d -> %d -> %d" % (len(frame1.feature_descriptors), len(matches), len(inliers), sum(inliers)))

    transformation_matrix = decomposeFundamentalMatrix(model.params)
    return keypoints1[inliers], keypoints2[inliers], transformation_matrix
