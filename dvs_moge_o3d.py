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




CAM_W, CAM_H = 700, 600
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
dt = 0.4
lamda = 0.1
mesh_path = "scenes/office2.glb"
img_path = "imgs/room.jpeg"



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



def move_points_with_camera(points_world, T_wc):
    # moves points in the WORLD frame so that: new_camera_coordinates == old_world_coordinates
    #which is done with a direct transformation
    H, W, _ = points_world.shape

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]

    Pw = points_world.reshape(-1, 3).T   # (3, N)

    # DIRECT transform (camera → world)
    Pw_new = R_wc @ Pw + t_wc[:, None]

    return Pw_new.T.reshape(H, W, 3)






def load_points(img_path, cam_homog) :

    # Load MoGe-2 model
    device = "cuda"
    model = MoGeModel.from_pretrained(
        "Ruicheng/moge-2-vitl-normal"
    ).to(device)
    model.eval()


    # Load RGB image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = torch.tensor(
        img / 255.0,
        dtype=torch.float32,
        device=device
    ).permute(2, 0, 1)


    # Run MoGe-2 inference and get points
    output = model.infer(image)
    points = output["points"]   # (H, W, 3(xyz))  metric 3D points
    mask   = output["mask"]     # (H, W, 1(Bool)) trusted pixels
    points = points.cpu().numpy()
    mask   = mask.cpu().numpy()

    #turn coords to cam frame
    points = move_points_with_camera(points, cam_homog)

    # Get colors of each pixel of img to color the points
    colors = img.astype(np.float32) / 255.0

    # Flatening things
    pts     = points.reshape(-1, 3)
    colors  = colors.reshape(-1, 3)
    valid   = mask.reshape(-1) > 0

    # Create Open3D point cloud with only valid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    #o3d.io.write_point_cloud("scenes/room_points.ply", pcd)
    return pcd



def set_cameras(cur_pose_vect) :

    # making athe world axis // default args keep position 000 matching wf nd also the size as 1 same as wf units
    des_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    cur_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Making the des_pose centered in front of the scene by transforming wf with 2 succssf trnsfs
    T_cw1 = get_homog([0, 0, 0, 0, -np.pi/2, 0])  
    T_cw2 = get_homog([0, 0, 0, 0, 0, np.pi])  
    T_cw3 = get_homog([5, -4, -20, 0, 0, 0])  
    T_cw4 = get_homog([0, 0, 0, 0, 0, 0])  
    des_T_cw = T_cw1 @ T_cw2 @ T_cw3 @ T_cw4
    des_T_cw = get_homog([0,0,0,0,0,0])
    des_camera_axis.transform(des_T_cw)

    # Making the cur_pose by transforming the des_pose with cur_pose_vect
    T_cw = get_homog(cur_pose_vect)
    cur_T_cw = des_T_cw @ T_cw
    cur_camera_axis.transform(cur_T_cw)

    return cur_T_cw, des_T_cw, des_camera_axis, cur_camera_axis





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




def compute_image_interaction_matrix(cur_depth_map, grad_Ix, grad_Iy):    

    # Create 2d of just u coords nd 2d of just v coords, both (H, W)
    u_coords, v_coords = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))  
    
    # get a 2d of x coords nd y coords in meters , both (H, W)
    x = (u_coords - CX) / FX
    y = (v_coords - CY) / FY
    
    # get a copy of depth with 0s as 100
    Z = cur_depth_map.copy()
    Z[Z == 0] = 100
    
    # Flatten everything for computation, from H,W to H*W
    x_flat = x.ravel() 
    y_flat = y.ravel() 
    Z_flat = Z.ravel() 
    grad_Ix_flat = grad_Ix.ravel() 
    grad_Iy_flat = grad_Iy.ravel() 


    # we will do − (Ix_Lx + Iy_Ly)
    # np.stack takes arrays of same shape nd stack nth element of each array in a single array then go to (n+1)th elemnt
    # this will give [[lx1], [lx2], [lx2].. ]

    Ix_Lx = grad_Ix_flat[:, None] * np.stack([
        -f / Z_flat,                   
        np.zeros_like(Z_flat),          
        x_flat / Z_flat,                
        x_flat * y_flat / f,            
        -(f + (x_flat * x_flat) / f),   
        y_flat                          
    ], axis=1)  # (H*W, 6)
    
    Iy_Ly = grad_Iy_flat[:, None] * np.stack([
        np.zeros_like(Z_flat),          
        -f / Z_flat,                    
        y_flat / Z_flat,                
        f + (y_flat * y_flat) / f,      
        -(x_flat * y_flat) / f,         
        -x_flat                         
    ], axis=1)  # (H*W, 6)


    Ls = -(Ix_Lx + Iy_Ly)  # (H*W, 6)    
    return Ls




def scale_by_max(v, max_val):
    v = np.asarray(v, dtype=float)
    m = np.max(np.abs(v))
    if m == 0:
        return v
    return v * (max_val / m)




def update_cam_pose(cam_homog, Vc, dt): 
 
     # Getting w and v vector (expresed in Cf) and the homog_mat in Wf : T
     T = np.array(cam_homog, dtype=float) 
     vx, vy, vz, wx, wy, wz = [float(x) for x in Vc] 
     w = np.array([wx, wy, wz], dtype=float) 
     v = np.array([vx, vy, vz], dtype=float) 
 
     R_cur = T[:3, :3]  # Current rotation matrix in Wf
     t_cur = T[:3, 3]   # Current translation vector in Wf
     
     # Step 1: Update translation
     # v is trans veloc in cam frame, v*dt give the trans vector made by v*dt (still in cam frame ofc)
     v_cam = v * dt
     # turn this v_cam to wf so we can update our homog matrix
     v_world = R_cur @ v_cam
     # Update translation in world frame
     t_new = t_cur + v_world
     
     # Step 2: Update rotation separately
     w_norm = np.linalg.norm(w) 
     I3 = np.eye(3) 
     
     if w_norm < 1e-12: 
         # Rotation is very small, use first-order approximation (tangent space)
         # R_delta ≈ I + skew(w*dt)
         w_skew = LinAlgeb.skew(w * dt)
         R_delta = I3 + w_skew
     else: 
         # Use exponential map for larger rotations
         theta = w_norm * dt 
         u = w / w_norm 
 
         # Skew(u) 
         ux = LinAlgeb.skew(u) 
         
         # Rodrigues rotation formula
         R_delta = I3 + math.sin(theta)*ux + (1.0 - math.cos(theta))*(ux @ ux) 
     
     # Apply rotation update (body frame, so right-multiply)
     R_new = R_cur @ R_delta
     
     # Build new transformation matrix
     new_T = np.eye(4)
     new_T[:3, :3] = R_new
     new_T[:3, 3] = t_new
     
     return new_T




def get_grads_visp(I):
    # coefficients
    c1 = 2047.0
    c2 = 913.0
    c3 = 112.0
    norm = 8418.0

    H, W = I.shape
    I = I.astype(np.float64)

    # adding edges length 3 in all sides (they get the value of their neighbours)
    I_padded = np.pad(I, pad_width=3, mode='edge')

    # X derivative (horizontal)
    # we can divide the width indinces of the padded-img as follows: 0(1st elmnt of pad_img)-3(crspnd to 1st elmnt of orig-img)-H+3(crspnd to last elmnt of orig-img)-H+6(lst elmnt of pd-img)
    dI_du = (
        c1 * (I_padded[3:H+3, 4:W+4] - I_padded[3:H+3, 2:W+2]) +  # j+1 vs j-1
        c2 * (I_padded[3:H+3, 5:W+5] - I_padded[3:H+3, 1:W+1]) +  # j+2 vs j-2
        c3 * (I_padded[3:H+3, 6:W+6] - I_padded[3:H+3, 0:W+0])    # j+3 vs j-3
    ) / norm

    # Y derivative (vertical)
    dI_dv = (
        c1 * (I_padded[4:H+4, 3:W+3] - I_padded[2:H+2, 3:W+3]) +  # i+1 vs i-1
        c2 * (I_padded[5:H+5, 3:W+3] - I_padded[1:H+1, 3:W+3]) +  # i+2 vs i-2
        c3 * (I_padded[6:H+6, 3:W+3] - I_padded[0:H+0, 3:W+3])    # i+3 vs i-3
    ) / norm

    # turning grad values to frm pxls to cam units
    Ix = FX * dI_du
    Iy = FY * dI_dv

    return Ix, Iy




def save_img(img, title, folder_path) :
    img = np.asarray(img)
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{folder_path}/{title}.png", img_u8)




def initialize_traject_visualizer(vis, trajectory_line, des_cam_axis, cur_cam_axis, des_extrins, mesh) :

    # init visualizer 
    vis.create_window(
        width=800,
        height=800,
        window_name="Camera Trajectory Scene")
    vis.add_geometry(mesh, reset_bounding_box=True)
    vis.add_geometry(des_cam_axis,  reset_bounding_box=False)
    vis.add_geometry(cur_cam_axis, reset_bounding_box=False)
    vis.add_geometry(trajectory_line, reset_bounding_box=False)


    # Define the POV pose (static observer camera)
    T_cw = get_homog([3, -1, -3, 0, 0, 0])
    T_cw1 = get_homog([0, 0, 0, 0, 0, 0])
    T_cw2 = get_homog([0, 0, 0, 0, 0, 0])
    pov_extrins = des_extrins @ T_cw @ T_cw1 @ T_cw2
    pov_extrins = np.linalg.inv(pov_extrins)

    # Create Open3D camera parameters
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = intrins2
    cam_params.extrinsic = pov_extrins

    # Apply camera to visualizer
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        cam_params,
        allow_arbitrary=True
    )

    # set up things somehow hh
    vis.poll_events()
    #vis.update_renderer()




def update_traject_visualizer(i, vis,trajectory_line, camera_centers, cur_extrins, prev_camera_axis) :

    # Create current camera frame
    cur_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    cur_camera_axis.transform(cur_extrins)

    # Camera center (trajectory point)
    center = np.array(cur_camera_axis.get_center()).reshape(1, 3)
    camera_centers.append(center)

    # Update trajectory
    points = np.vstack(camera_centers)
    if len(points) > 1:
        lines = np.array([[i, i + 1] for i in range(len(points) - 1)])
    else:
        lines = np.zeros((0, 2))
    trajectory_line.points = o3d.utility.Vector3dVector(points)
    trajectory_line.lines = o3d.utility.Vector2iVector(lines)
    trajectory_line.paint_uniform_color([1, 0, 0])

    # Replace previous camera frame
    try:
        vis.remove_geometry(prev_camera_axis, reset_bounding_box=False)
    except NameError:
        pass

    vis.add_geometry(cur_camera_axis, reset_bounding_box=False)
    prev_camera_axis = cur_camera_axis

    # Update visualizer ONLY (no POV change)
    vis.update_geometry(trajectory_line)
    vis.update_geometry(cur_camera_axis)
    vis.poll_events()
    #vis.update_renderer()

    img = vis.capture_screen_float_buffer()
    #save_img(img, i, "traject")

    return prev_camera_axis, img



def compute_grayscale_difference(img1, img2, normalize=True):
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    diff = np.abs(img1_float - img2_float)
    
    if normalize:
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        else:
            diff = diff.astype(np.uint8)
    else:
        diff = diff.astype(np.uint8)
    
    if diff.ndim == 2:
        diff = diff[:, :, np.newaxis]
    
    return diff








#____________________________________________________________________________________


def main() :

    
    # Load mesh nd setup 
    #mesh = load_mesh(mesh_path, lambert=False)
    
    # setup init nd desired homog matrices nd axises
    cur_pose_vect = [2, 0, -2, 0, 0, 0]  
    cur_extrins, des_extrins, des_cam_axis, cur_cam_axis = set_cameras(cur_pose_vect)

    # get init 3d points from img with moge nd translate them to cur_cam frame
    points = load_points(img_path, cur_extrins)
    mesh = points

    
    # taking pic from des_cam
    des_img, des_depth = takin_pic(mesh, des_extrins)
    #plot_img(des_img, "desired_image")
    #plot_img(des_depth, "desired_depthMap")
    save_img(des_img, "des", "frames")
    gray_des = 0.299 * des_img[:, :, 0] + 0.587 * des_img[:, :, 1] + 0.114 * des_img[:, :, 2]
    S_star = gray_des.flatten()


    # intialize trajectory scene : the visualizer with frames, mesh and line
    camera_centers = []
    trajectory_line = o3d.geometry.LineSet()
    o3d_vis = o3d.visualization.Visualizer()
    initialize_traject_visualizer(o3d_vis, trajectory_line, des_cam_axis, cur_cam_axis, des_extrins, mesh)
    prev_cur_frame = None




    #____________________________________________________________________________________

    # Closed-loop
    try:
        for i in range(999):

            # 0 - Update visualizer of trajects
            prev_cur_frame, scene_img = update_traject_visualizer(i, o3d_vis, trajectory_line, camera_centers, cur_extrins, prev_cur_frame)

            # 1 - Capture current img nd get S
            cur_img, cur_depth_map = takin_pic(mesh, cur_extrins)
            save_img(cur_img, i, "frames")
            gray_cur = 0.299 * cur_img[:, :, 0] + 0.587 * cur_img[:, :, 1] + 0.114 * cur_img[:, :, 2]
            S = gray_cur.flatten()

            # make our matplotlib vis
            if(i==0) :
                matp_vis = LiveOptimizationVisualizer(des_img, cur_img)


            # 2 - Compute the cost and the diff img for visua
            diff = S - S_star
            cost = diff.T @ diff
            print(f"Cost {i} :", cost)
            current_diff_img = compute_grayscale_difference(gray_cur, gray_des)

            # 3 - Compute Gradient and Ls 
            grad_Ix, grad_Iy = get_grads_visp(gray_cur)
            Ls = compute_image_interaction_matrix(cur_depth_map, grad_Ix, grad_Iy)

            # 4- scaling the v based on our pose

            
            if cost > 10000 :  
                max_val = 0.1 
                mu = 999999   
            elif cost > 6000 :  
                max_val = 0.1 
                mu=999999 
            elif cost > 500 :  
                max_val = 0.01  
                mu=0.0000001 
            elif cost > 50 :  
                max_val = 0.001  
                mu=0.0000001 
            elif cost > 15 : 
                max_val = 0.001  
            else : 
                break 
                    

            # 5 - Compute V with GN or LM 
            V = -lamda * np.linalg.solve(Ls.T @ Ls + mu * np.diag(np.diag(Ls.T @ Ls)), Ls.T @ diff)      
            #V = -lamda * np.linalg.pinv(Ls) @ diff
            V = scale_by_max(V, max_val)

            
            # 6 - Update camera pose & update matplotlib vis data
            cur_extrins = update_cam_pose(cur_extrins, V, dt)
            matp_vis.update(i, scene_img, cur_img, current_diff_img, V,  cost)

        matp_vis.close()
        o3d_vis.close()

    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting loop cleanly.")
        matp_vis.close()
        o3d_vis.close()
        os._exit(0)



if __name__ == "__main__":
    main()