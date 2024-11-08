import numpy as np
import gtsam
from utils import *
import cv2


def optimize_poses_and_points(poses, map_points, K, observability_graph):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    projection_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1]))
    K_gtsam = gtsam.Cal3_S2(K[0, 0], K[1, 1], 1, K[0, 2], K[1, 2])

    for frame_id, pose in poses.items():
        pose_key = gtsam.symbol('p', frame_id)
        pose = gtsam.Pose3(gtsam.Rot3.Rodrigues(pose[0], pose[1], pose[2]), pose[3:])
        pose = pose.inverse()
        initial_estimate.insert(pose_key, pose)
    
    fixedPose = gtsam.Pose3()
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-5)
    graph.add(gtsam.PriorFactorPose3(gtsam.symbol('p', 0), fixedPose, priorNoise))

    
    for point_index, observations in observability_graph.items():
        landmark_key = gtsam.symbol('l', point_index)
        # TODO: добавьте констрейнты для репроекций
        # см. GenericProjectionFactorCal3_S2

    print ("====refinement=====")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    print ("initial error: ", optimizer.error())
    optimized_values = optimizer.optimize()
    print ("took ", optimizer.iterations(), "iterations")
    print ("final error: ", optimizer.error())
    print ("==================")
    updated_poses = {}

    for frame_id in poses.keys():
        optimized_pose = optimized_values.atPose3(gtsam.symbol('p', frame_id)).inverse()
        rvec = rotation_to_lie(optimized_pose.rotation().matrix())
        t = optimized_pose.translation()
        updated_poses[frame_id] = np.concatenate((rvec.squeeze(), t.squeeze()))

    updated_map_points = map_points.copy()
    for point_index in observability_graph.keys():
        if (optimized_values.exists(gtsam.symbol('l', point_index))):
            updated_map_points[point_index] = optimized_values.atVector(gtsam.symbol('l', point_index))

    return updated_poses, updated_map_points