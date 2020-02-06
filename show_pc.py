import open3d as o3d
import argparse

parser = argparse.ArgumentParser(
        description=
        "Simple Script to show a point cloud")
parser.add_argument("--path2ply",
                    default="data3/scene/integrated.ply",
                    help="choose a ply file")

args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.path2ply)

o3d.visualization.draw_geometries([pcd])