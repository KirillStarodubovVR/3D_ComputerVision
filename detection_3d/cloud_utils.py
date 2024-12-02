import torch
import numpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import open3d as o3d
import numpy as np

INDICES_1 = [5,4,0,1]
INDICES_2 = [6,5,1,2]
INDICES_3 = [7,6,2,3]
INDICES_4 = [4,7,3,0]

PCD_SCENE=dict(
        xaxis=dict(visible=False,range=[-100,100]),
        yaxis=dict(visible=False,range=[-100,100]),
        zaxis=dict(visible=False,),
        aspectmode='data'
)


class PointCloud():
    nbr_dims = 4

    def __init__(self, points : np.ndarray) -> None:

        assert points.shape[1] == PointCloud.nbr_dims, 'Error: Pointcloud points must have format: n x {}'.format(PointCloud.nbr_dims)
        self.points = points.copy()

    def translate(self, x: np.ndarray) -> None:
        self.points[:, :3] += x[:3]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        self.points[:, :3] = np.dot(rot_matrix, self.points[:, :3].T).T

    @staticmethod
    def from_file(file_name: str):
        if file_name.endswith('.bin'):
            scan = np.fromfile(file_name, dtype=np.float32)
            points = scan.reshape((-1, 4))
        elif file_name.endswith('.pcd'):
            cloud = o3d.t.io.read_point_cloud(file_name)
            points = np.concatenate((cloud.point['positions'].numpy(), cloud.point['intensity'].numpy()), axis=1)
        else:
            raise ValueError("Unsupported point cloud file type")
        return PointCloud(points)


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def get_scatter3d_plot(x,y,z, mode='lines', marker_size=1, color=None, opacity=1, colorscale=None, **kwargs):
    return go.Scatter3d(x=x, y=y, z=z, mode=mode, hoverinfo='skip',showlegend=False, 
                        marker = dict(size=marker_size, color=color, opacity=opacity, colorscale=colorscale), **kwargs)

def plot_pc_data3d(x,y,z, apply_color_gradient=True, color=None, marker_size=1, colorscale=None, **kwargs):
    if apply_color_gradient:
        color = np.sqrt(x**2 + y **2 + z **2)
    return get_scatter3d_plot(x,y,z, mode='markers', color=color, colorscale=colorscale, marker_size=marker_size, **kwargs)

def plot_box_corners3d(box3d, color,**kwargs):
    return [
        get_scatter3d_plot(box3d[INDICES_1, 0], box3d[INDICES_1, 1], box3d[INDICES_1, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_2, 0], box3d[INDICES_2, 1], box3d[INDICES_2, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_3, 0], box3d[INDICES_3, 1], box3d[INDICES_3, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_4, 0], box3d[INDICES_4, 1], box3d[INDICES_4, 2], color=color, **kwargs),
    ]


def plot_bboxes_3d(boxes3d, box_colors, **kwargs):
    # boxes3d shape = (N,8,3) = bounding box corners in 3d coordinates
    # box_colors = (N) length vector
    boxes3d_objs = []
    for obj_i in range(boxes3d.shape[0]):
        boxes3d_objs.extend(plot_box_corners3d(boxes3d[obj_i], color = box_colors[obj_i], **kwargs))
    return boxes3d_objs

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def get_voxel_corners(voxel_indices, voxel_size, pc_range):
    voxel_centres = (voxel_indices * voxel_size) + pc_range[0:3] + (voxel_size * 0.5)

    voxel_bboxes = np.column_stack((voxel_centres, np.repeat( np.append(voxel_size, 0.0)[None,:], len(voxel_indices), axis=0)))
    voxel_corners = boxes_to_corners_3d(voxel_bboxes)
    return voxel_corners

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


