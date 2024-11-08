import cv2
import numpy as np
"""

"""
def triangulate_new_points(frame1, frame2, K, Tw_f1, Tw_f2, feature_coverage = None):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)
    #удаляем из 2-ого изображения фичи, 
    #которые задетектились рядом с существующими фичами
    if feature_coverage is not None or False:
        h, w = feature_coverage.shape
        rY = frame1.shape[0] // h
        rX = frame1.shape[1] // w
        filtered_features = [i for i, kp in enumerate(kp2) 
                             if feature_coverage[int(kp.pt[1]) // rY,
                                                int(kp.pt[0]) // rX] == 0]
        kp2 = [kp2[i] for i in filtered_features]
        des2 = des2[filtered_features, :]
    
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m in matches if m[0].distance < 0.7 * m[1].distance]

    if not good_matches:
        return np.empty((0, 3)), np.empty((0, 128))

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    P1 = K @ Tw_f1
    P2 = K @ Tw_f2
    points_4D = cv2.triangulatePoints(P1, P2, src_pts.reshape(-1, 2).T, dst_pts.reshape(-1, 2).T)
    points_3D = points_4D / points_4D[3]

    # проверяем, что точки триангулировались корректно 
    # (т.е не соответвуют "мусорным" матчам)
    depths2 = (Tw_f2[:, :3] @ points_3D[:3, :] + Tw_f2[:, 3].reshape(3, 1))[2]
    
    #глубина должны быть положительная
    mask = depths2 > 0

    projected_pts1 = (P1 @ points_4D)[:2] / (P1 @ points_4D)[2]
    projected_pts2 = (P2 @ points_4D)[:2] / (P2 @ points_4D)[2]

    reprojection_error1 = np.sum((src_pts.reshape(-1, 2) - projected_pts1.T)**2, axis=1)
    reprojection_error2 = np.sum((dst_pts.reshape(-1, 2) - projected_pts2.T)**2, axis=1)
    error_threshold = 5  

    # точки должны репроецироваться в задетекченные фичи
    mask = (mask & (reprojection_error1 < error_threshold) &
            (reprojection_error2 < error_threshold)) 
    
    print ("Added", mask.sum(), "points", sep=" ")

    descriptors = np.array([des2[m[0].trainIdx] for m in good_matches])

    points_3D = points_3D[:, mask]
    descriptors = descriptors[mask, :]
    src_pts = src_pts[mask, :, :]
    dst_pts = dst_pts[mask, :, :]

    return points_3D[:3].T, descriptors, src_pts, dst_pts