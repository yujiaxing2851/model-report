import trimesh
import numpy as np
from scipy.spatial import cKDTree
import pye57
import open3d as o3d
import igl
import pywavefront
import pandas as pd


def read_e57_file(file_path):
    # 读取 .e57 文件
    e57 = pye57.E57(file_path)
    # header = e57.get_header(0)
    # print([x for x in header])
    # data3D = e57.read_scan(0, intensity=True, colors=True, row_column=True)
    data3D = e57.read_scan(0, colors=True, ignore_missing_fields=True)
    # print(header.point_count)
    # print(header.rotation_matrix)
    # print(header.translation)

    # data3D.keys() are:
    # cartesianX cartesianY cartesianZ
    # intensity
    # colorRed colorGreen colorBlue
    # rowIndex columnIndex

    positions = np.vstack((data3D['cartesianX'], data3D['cartesianY'], data3D['cartesianZ'])).transpose()
    if 'colorRed' in data3D and 'colorGreen' in data3D and 'colorBlue' in data3D:
        colors = (
            np.vstack((data3D['colorRed'], data3D['colorGreen'], data3D['colorBlue'])).transpose() / 255.0
        )  # Normalize to [0, 1]
        print(np.max(colors, axis=0))
    else:
        colors = np.zeros_like(positions)  # 如果没有颜色数据，使用零填充

    return positions, colors

def load_obj_by_groups_manual_clean(obj_path):
    meshes = []
    current_name = None
    current_faces = []
    vertices = []

    with open(obj_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = tuple(float(x) for x in parts[1:])
                vertices.append(vertex)

            elif line.startswith('o '):
                # 保存前一个物体
                if current_name is not None and current_faces:
                    mesh = build_clean_mesh(vertices, current_faces)
                    meshes.append(mesh)
                    current_faces = []

                current_name = line.strip().split(maxsplit=1)[1]

            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                current_faces.append(face)

    # 处理最后一个物体
    if current_name is not None and current_faces:
        mesh = build_clean_mesh(vertices, current_faces)
        meshes.append(mesh)
    print(len(meshes))

    return meshes

def build_clean_mesh(all_vertices, faces):
    # 提取所有面用到的顶点索引
    used_indices = set(i for face in faces for i in face)
    used_indices = sorted(used_indices)

    # 建立旧索引到新索引的映射
    index_mapping = {old: new for new, old in enumerate(used_indices)}

    # 提取用到的顶点
    new_vertices = np.array([all_vertices[i] for i in used_indices])

    # 重新编号 faces
    new_faces = np.array([[index_mapping[i] for i in face] for face in faces])

    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)



def resample_mesh(mesh, count=10000):
    return mesh.sample(count)

# def assign_point_to_named_meshes(point_cloud, mesh_name_pairs, samples_per_mesh=10000):
#     """为每个点分配到最近的命名 mesh"""
#     sampled_points = []
#     sampled_names = []

#     for mesh, name in mesh_name_pairs:
#         points = resample_mesh(mesh, count=samples_per_mesh)
#         sampled_points.append(points)
#         sampled_names.extend([name] * len(points))

#     all_samples = np.vstack(sampled_points)
#     kd_tree = cKDTree(all_samples)

#     dists, indices = kd_tree.query(point_cloud, k=1)
#     object_names = [sampled_names[i] for i in indices]

#     return dists, object_names

def assign_point_to_mesh_indices(point_cloud, mesh_name_pairs, samples_per_mesh=10000):
    """为每个点分配到最近的 mesh 下标"""
    sampled_points = []
    sampled_indices = []


    for idx, (mesh, name) in enumerate(mesh_name_pairs):
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue  # 跳过空 mesh
        points = resample_mesh(mesh, count=samples_per_mesh)
        sampled_points.append(points)
        sampled_indices.extend([idx] * len(points))  # 用索引代替名称

    # 所有采样点合并为一个大数组
    all_samples = np.vstack(sampled_points)
    sampled_indices = np.array(sampled_indices)

    # 构建 KDTree
    kd_tree = cKDTree(all_samples)

    # 查询每个点最近的 mesh 样本点
    dists, indices = kd_tree.query(point_cloud, k=1)

    # 返回对应下标
    object_indices = sampled_indices[indices]

    return dists, object_indices


# 主流程
if __name__ == "__main__":
    wrong_models = pd.read_excel('data/wrong_models.xlsx', sheet_name='Sheet1')

    obj_path = 'data/All plant.obj'  # 替换为你的文件路径
    mesh = load_obj_by_groups_manual_clean(obj_path)
    
    p1, _ = read_e57_file('data/BASF_860_LEVEL1.e57')
    p2, _ = read_e57_file('data/BASF_860_LEVEL2.e57')
    p3, _ = read_e57_file('data/BASF_860_LEVEL3.e57')

    # 合并点云
    P = np.vstack((p1, p2, p3))

    

    indices = np.load('output/object_indices.npy')

    wrong_id = wrong_models['Model_ID'].to_list()

    print(len(wrong_id))


    selected_meshes = [mesh[i] for i in wrong_id]

    print(len(selected_meshes))

    # 合并为一个 mesh
    scene = trimesh.Scene()
    for i, mesh in enumerate(selected_meshes):
        if isinstance(mesh, trimesh.Trimesh):
            scene.add_geometry(mesh, node_name=f'mesh_{i}')
        else:
            print(f"Warning: selected_meshes[{i}] is not a Trimesh object, it is {type(mesh)}")

    # 导出为 .obj（会自动合并为一个 obj 文件）
    scene.export('output/wrong_meshes.obj')

    wrong_id_set = set(wrong_id)

    indices_in_I = np.where(np.isin(indices, list(wrong_id_set)))[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[indices_in_I])

    # 保存为 PLY 或 PCD
    o3d.io.write_point_cloud("output/wrong_meshes_holding_points.pcd", pcd)






