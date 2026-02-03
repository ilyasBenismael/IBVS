# correct gspat rast prob
# Apply dvs and ibvs on diff params(see limits and all)

# Then do use these posed images and sfm =>  import all in o3d and visualize => Apply MoGe on all of them and get moge points and put the poses up to metric scale => initialize 
# The gaussians and apply IBVS (make sure things are clean and minimal, get inspired but other paper initializations)
# Study well the weak points and ibvs behavior

# Try the incremental SFM-GS with IBVS (BigStudy, then study other papers and other repositories and try applying improvements and studying weak points..)








import open3d as o3d
import numpy as np
from utils.lin_algeb import LinAlgeb
import cv2
import matplotlib.pyplot as plt
import math
import os
from PIL import Image
from datetime import datetime
from utils.visualizer import LiveOptimizationVisualizer
import torch
from moge.model.v2 import MoGeModel
import pycolmap









#IBVS params
dt = 0.4
lamda = 0.1

#Scene params
mesh_path = "scenes/office2.glb"
img_path = "imgs/room.jpeg"


#Robot's camera camera 
CAM_W, CAM_H = 1500, 1000
FX = FY = 0.8 * max(CAM_W, CAM_H)
f=1
CX, CY = CAM_W / 2.0, CAM_H / 2.0
intrins = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W,
    height=CAM_H,
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY
)


#3rd person viewer camera 
CAM_W2, CAM_H2 = 400, 300
FX2 = FY2 = 0.8 * max(CAM_W2, CAM_H2)
CX2, CY2 = CAM_W2 / 2.0, CAM_H2 / 2.0
intrins2 = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W2,
    height=CAM_H2,
    fx=FX2,
    fy=FY2,
    cx=CX2,
    cy=CY2
)


#some configs 
np.set_printoptions(precision=2, suppress=False)






def get_homog(pose_vector) :
    R,t = LinAlgeb.make_rot_trans(*pose_vector)
    return LinAlgeb.get_homog_matrix(R,t)





def load_mesh(path, lambert) :
    
    # loading the mesh, enable post to render the colored texture, (no normals for lambertian)
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    print(f"Number of vertices: {len(mesh.vertices)}")

    
    if(not lambert) :
        mesh.compute_vertex_normals()  

    # center the mesh, and scaling it
    mesh.translate(-mesh.get_center()) 
    mesh.scale(3 , center=mesh.get_center()) 

    return mesh 






def set_init_cam() :

    # making athe world axis // default args keep position 000 matching wf nd also the size as 1 same as wf units
    init_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

    # Making the des_pose centered in front of the scene by transforming wf with 2 succssf trnsfs
    T_cw1 = get_homog([0, 0, 0, 0, 0, -np.pi])  
    T_cw2 = get_homog([0, 0, 0, 0, np.pi/2, 0])  
    T_cw3 = get_homog([0, 0, -10, 0, 0, 0])
    init_extrins = T_cw1 @ T_cw2 @ T_cw3
    init_cam_axis.transform(init_extrins)

    return init_extrins, init_cam_axis





def visualize_scene(scene_compos) :
        o3d.visualization.draw_geometries(
        scene_compos,
        window_name="The scene",
        width=800,
        height=800,
        mesh_show_back_face=True)


    
def takin_pic(mesh, extrins):

    # defining extrins and T_cw_final params 
    extrins = np.linalg.inv(extrins)

    # making our pincamparams objct
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = intrins
    cam_params.extrinsic = extrins

    # ---------- Create visualizer ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=CAM_W, height=CAM_H, visible=False)
    vis.add_geometry(mesh)


    # ---------- Apply camera ----------
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        cam_params,
        allow_arbitrary=True
    )

    # ---------- Render once ----------
    vis.poll_events()
    vis.update_renderer()

    # ---------- Capture image ----------
    img = vis.capture_screen_float_buffer()
    depth = vis.capture_depth_float_buffer()
    vis.destroy_window()

    return np.asarray(img), np.asarray(depth) 




def plot_img(des_img, title) :
    plt.imshow(des_img)
    plt.axis("off")
    plt.suptitle(title)
    plt.show()




def save_img(img, title, folder_path) :
    img = np.asarray(img)
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{folder_path}/{title}.png", img_u8)




def get_init_axes_and_imgs(
    extrins1,
    scene_mesh,
    motions,
    compos,
    img_prefix="img",
    save_dir="frames",
    axis_size=0.5,
):

    for i, m in enumerate(motions, start=2):

        # build transform
        T = (
            extrins1
            @ get_homog([*m["t"], 0, 0, 0])
            @ get_homog([0, 0, 0, *m["r"]])
        )

        # take image
        img, depth = takin_pic(scene_mesh, T)
        save_img(img, f"{img_prefix}{i}", save_dir)

        # create axis
        cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size
        )

        # visualize camera in world → invert if T is world->camera
        cam_axis.transform(np.linalg.inv(T))

        compos.append(cam_axis)









#___________________________________________________________________________________________________________




def main() :
    
    # Load mesh nd setup 
    scene_mesh = load_mesh(mesh_path, lambert=False)

    # Setup init nd desired homog matrices nd axises
    extrins1, init_cam_axis = set_init_cam()
    visualize_scene([scene_mesh, init_cam_axis])
    compos = [init_cam_axis]
    
    # Taking pic from initial pose
    img1, des_depth = takin_pic(scene_mesh, extrins1)
    plot_img(img1, "img1")
    save_img(img1, "img1", "frames")


    # taking 10 init randome img
    motions = [
    {"t": [2, -1, 1],  "r": [0, -np.pi/10, 0]},
    {"t": [4, 0, 0],   "r": [0, -np.pi/10, 0]},
    {"t": [6, 1, 1],   "r": [0, -np.pi/8,  0]},
    {"t": [8, 0, 0],   "r": [0, -np.pi/8,  0]},
    {"t": [10, -1, 1], "r": [0, -np.pi/8,  0]},
    {"t": [12, 0, 0],  "r": [0, -np.pi/6,  0]},
    {"t": [14, 1, 1],  "r": [0, -np.pi/6,  0]},
    {"t": [16, 0, 0],  "r": [0, -np.pi/5,  0]},
    {"t": [18, -1, 1], "r": [0, -np.pi/5,  0]},]
    
    
    get_init_axes_and_imgs(extrins1=extrins1,
    scene_mesh=scene_mesh,
    motions=motions,
    compos=compos)


    recon = pycolmap.Reconstruction("/home/user/Bureau/mini_sfm1/sparse/0")


    homog_poses = {}
    for image in recon.images.values():
        if not image.has_pose:
            print(image.name)
            continue
        T_cw = image.cam_from_world()   # Rigid3d
        R = T_cw.rotation.matrix()      # (3,3)
        t = T_cw.translation            # (3,)
        T_h = LinAlgeb.get_homog_matrix(R,t)
        homog_poses[image.name] = T_h



    for img_name, T in homog_poses.items():
        # If T is world->camera, invert it
        T_vis = np.linalg.inv(T)
        cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
        cam_axis.transform(T_vis)
        compos.append(cam_axis)

    visualize_scene(compos)    



   










if __name__ == "__main__":
    main()