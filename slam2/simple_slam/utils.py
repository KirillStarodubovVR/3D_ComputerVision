import cv2
import numpy as np

# For visualization

def draw_trajectory(window_name, poses_map):
    TRAJ_SIZE = 1000
    CENTER_OFFSET = TRAJ_SIZE // 2
    SCALE = 1.5
    poses = list(poses_map.values())
    traj = np.zeros((TRAJ_SIZE, TRAJ_SIZE, 3), dtype=np.uint8)

    for i in range(len(poses)):
        x_position = int(CENTER_OFFSET + SCALE * poses[i][3])
        z_position = int(CENTER_OFFSET + SCALE * poses[i][5])

        cv2.circle(traj, (x_position, z_position), 3, (0, 0, 255), 2)

        if i > 0:
            x_previous = int(CENTER_OFFSET + SCALE * poses[i-1][3])
            z_previous = int(CENTER_OFFSET + SCALE * poses[i-1][5])
            cv2.line(traj, (x_previous, z_previous), (x_position, z_position), (0, 255, 0), 2)
            
    cv2.imshow(window_name, traj)
    cv2.waitKey(1)

def rotation_to_lie(R):
    # Calculate the rotation angle
    theta = np.arccos((np.trace(R) - 1) / 2)
    # If the rotation angle is small, avoid division by a small number
    if np.isclose(theta, 0):
        return np.array([0, 0, 0])
    # Compute the skew-symmetric matrix
    w_skew = theta / (2 * np.sin(theta)) * (R - R.T)
    # Extract the coordinates from the skew-symmetric matrix
    w = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]])
    return w