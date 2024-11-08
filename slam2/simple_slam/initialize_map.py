import cv2
import numpy as np

from utils import *
from triangulation import triangulate_new_points

def initialize_map(frame1, frame2, K):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m1 for m1, m2 in matches if m1.distance < 0.6 * m2.distance]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)

    map_points, descriptors, src_points, dst_points  = triangulate_new_points(
        frame1, frame2, K, np.eye(3, 4), np.hstack((R, t)))
 

    observability_graph = {}
    for i in range(map_points.shape[0]):
        observability_graph[i] = {0: src_points[i],
                                   1: dst_points[i]}

    poses = {}
    poses[0] = np.zeros(6)
    poses[1] = np.concatenate((rotation_to_lie(R), t.squeeze()))
    return poses, map_points, descriptors, observability_graph
