import cv2
import numpy as np

from utils import *
from initialize_map import initialize_map
from triangulation import triangulate_new_points
from refinement import optimize_poses_and_points

# Calibration matrix
K = np.array([
    [448.155164329302, 0, 640.0],
    [0, 448.155164329302, 360.0],
    [0, 0, 1]
])

observability_graph = {}  # A dictionary to store feature observations across frames

def get_feature_coverage_map(frame, n, projected_points):
    """Compute the feature coverage map for a frame."""
    h, w = frame.shape[:2]
    grid_h, grid_w = h // n, w // n
    coverage_map = np.zeros((n, n), dtype=np.uint8)
    for pt in projected_points:
        i, j = int(pt[1] // grid_h), int(pt[0] // grid_w)
        coverage_map[i, j] = 1
    return coverage_map

def visualize_coverage_map(coverage_map):
    """Visualize the feature coverage map."""
    enlarged_map = cv2.resize(coverage_map.astype(np.uint8)*255, (coverage_map.shape[1]*10, coverage_map.shape[0]*10), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Feature Coverage Map", enlarged_map)
    cv2.waitKey(1)

def update_map(frame, last_frame, map_points, active_points, map_des, K, observability_graph, poses, frame_count):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(frame, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(des, map_des[active_points], k=2)
    #TODO: чтобы пройти поворот, попробуйте матчить точки только в окрестости
    #детекта на последнем кадре (см. ORB SLAM)
    good_matches = [m1 for m1, m2 in matches if m1.distance < 0.5 * m2.distance]
    obj_pts = np.float32([map_points[active_points[m.trainIdx]] for m in good_matches])
    img_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, None)
    inliers = inliers.reshape(-1).tolist()
    good_matches = [good_matches[i] for i in inliers]
    img_pts = img_pts[inliers, :]
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts[inliers, :], img_pts, K, None, rvec, tvec)

    print ("Pnp Matches: ", len(good_matches))

    coverage = get_feature_coverage_map(last_frame, 20, img_pts.reshape(-1, 2))
    visualize_coverage_map(coverage)
   
    for i, m in enumerate(good_matches):
        if m.trainIdx not in observability_graph:
            observability_graph[active_points[m.trainIdx]] = {}
        observability_graph[active_points[m.trainIdx]][frame_count] = kp[m.queryIdx].pt
  
    active_points = [active_points[m.trainIdx] for m in good_matches]
    map_des[active_points, :] = np.float32([des[m.queryIdx] for m in good_matches])


    R, _ = cv2.Rodrigues(rvec)
    l_k = max(poses.keys())
    last_pose = poses[l_k]
    R_last, _ = cv2.Rodrigues(last_pose[:3].reshape(3, 1))
    t_last = last_pose[3:].reshape(3, 1)

    new_points_3D, new_des, src_points, dst_points = triangulate_new_points(
        last_frame, frame, K, np.hstack((R_last, t_last)), np.hstack((R, tvec)), coverage)
    for i in range(new_points_3D.shape[0]):
        observability_graph[len(map_points) + i] = {l_k: src_points[i], frame_count: dst_points[i]}
    active_points = active_points + [map_points.shape[0] + i for i in range(new_points_3D.shape[0])]
    print ("Active points: ", len(active_points))

    feature_img = frame.copy()
    for i in range(img_pts.shape[0]):
        cv2.circle(feature_img, img_pts[i,:].reshape(-1).astype(int), 5, (255, 0, 0), -1)
    for i in range(dst_points.shape[0]):
        cv2.circle(feature_img, dst_points[i,:].reshape(-1).astype(int), 5, (0, 255, 0), 1)
   
    cv2.imshow("Current Frame", feature_img)
    map_points = np.vstack((map_points, new_points_3D))
    map_des = np.vstack((map_des, new_des))
    return map_points, map_des, active_points, rvec, tvec, observability_graph

    


def simple_slam(video_file, K):
    cap = cv2.VideoCapture(video_file)
    # for i in range(270):
    #     cap.read()
    _, frame1 = cap.read()
    for i in range(3):
        cap.read()
    _, frame2 = cap.read()

    poses, map_points, map_des, observability_graph = initialize_map(frame1, frame2, K)
    #все точки активные после инициалицазии
    active_points = [i for i in range(map_points.shape[0])]
    frame_count = 5
    while True:
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            break

        if frame_count % 3 == 0 or frame_count > 280 and frame_count < 330:
            last_frame = frame2
            frame2 = frame
            map_points, map_des, active_points, rvec, tvec, observability_graph = update_map(frame, last_frame, map_points, active_points, map_des, K, observability_graph, poses, frame_count)

            poses[frame_count] = np.concatenate((rvec.squeeze(), tvec.squeeze()))
            draw_trajectory("Trajectory", poses)
            # cv2.waitKey(0)
            if len(poses) % 10 == 0:
                poses, map_points = optimize_poses_and_points(poses, map_points, K, observability_graph)

    cap.release()
    cv2.destroyAllWindows()

simple_slam('images.mkv', K)
