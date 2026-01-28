import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Sequence, List, Union
from matplotlib import cm
import cv2
import time
import open3d as o3d
from lin_algeb import LinAlgeb
import math
import os
from visualizer import LiveOptimizationVisualizer
from datetime import datetime
import torch
from moge.model.v2 import MoGeModel




# ------------- constants

CAM_W, CAM_H = 1656, 828 
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


mesh_path = "scenes/office2.glb"
img_path = "indoor.jpeg"



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






def get_Ss_from_points(points_in_cam: List[Sequence[float]]) -> List[List[float]]:
    s = []
    for p in points_in_cam :
        X, Y, Z = p
        if Z == 0 or Z < 0 :
            Z = 0.5
        x = f * (X / Z)
        y = f * (Y / Z)
        s.append([x, y])
    return s



def get_uv_from_Ss(Ss):
    uv_list = []
    for x, y in Ss:
        u = CX + FX * x 
        v = CY + FY * y
        u = int(np.clip(round(u), 0, CAM_W - 1))
        v = int(np.clip(round(v), 0, CAM_H - 1))
        uv_list.append([u, v])
    return uv_list




def get_homog(pose_vector) :
    R,t = LinAlgeb.make_rot_trans(*pose_vector)
    return LinAlgeb.get_homog_matrix(R,t)



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
    old_points_world = output["points"]   # (H, W, 3(xyz))  metric 3D points
    mask   = output["mask"]     # (H, W, 1(Bool)) trusted pixels
    old_points_world = old_points_world.cpu().numpy()
    mask   = mask.cpu().numpy()

    # translate points in front of current frame, the old world points coords are now the points coords in cam
    points_world = move_points_with_camera(old_points_world, cam_homog)
    points_cam = old_points_world

    # Get colors of each pixel of img to color the points
    colors = img.astype(np.float32) / 255.0

    # Flatening things
    pts     = points_world.reshape(-1, 3)
    colors  = colors.reshape(-1, 3)
    valid   = mask.reshape(-1) > 0

    # Create Open3D point cloud with only valid points
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(pts[valid])
    points_o3d.colors = o3d.utility.Vector3dVector(colors[valid])
    return points_o3d, points_world, points_cam



def set_cameras(cur_pose_vect) :

    # making athe world axis // default args keep position 000 matching wf nd also the size as 1 same as wf units
    des_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    cur_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Making the des_pose centered in front of the scene by transforming wf with 2 succssf trnsfs
    T_cw1 = get_homog([-15, 233, 44, 0, 0, 0])  
    T_cw2 = get_homog([0, 0, 0, 0, 0, 0])  
    T_cw3 = get_homog([0, 0, 0, 0, 0, 0])  
    T_cw4 = get_homog([0, 0, 0, 0, 0, 0])  
    des_T_cw = T_cw1 @ T_cw2 @ T_cw3 @ T_cw4
    des_camera_axis.transform(des_T_cw)

    # Making the cur_pose by transforming the des_pose with cur_pose_vect
    T_cw = get_homog(cur_pose_vect)
    cur_T_cw = des_T_cw @ T_cw
    cur_camera_axis.transform(cur_T_cw)

    return cur_T_cw, des_T_cw, des_camera_axis, cur_camera_axis



    
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




def get_sift_features(img):

    # Convert normalized RGB to uint8 format
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Convert RGB to grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des




def get_matches(nbr_features, kp1, des1, kp2, des2, cur_img, des_img, show) :

    cur_img_uint8 = (cur_img * 255).astype(np.uint8)
    des_img_uint8 = (des_img * 255).astype(np.uint8)
    
    # Match features using BFMatcher with L2 norm (for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (best matches first) and keep top 4
    matches = sorted(matches, key=lambda x: x.distance)[:nbr_features]

    if len(matches) < nbr_features:
        print(f"got just {len(matches)} matches") 
        #ValueError(f"Expected exactly 4 matches, but got {len(matches)}")
    
    cur_features = []
    des_features = []
    
    # Each match got the query index (index of kp from img1) and its corresp train index (index of kp from img2)
    for m in matches:
        idx1 = m.queryIdx
        idx2 = m.trainIdx
        (x1, y1) = kp1[idx1].pt
        cur_features.append([int(x1), int(y1)])
        (x2, y2) = kp2[idx2].pt
        des_features.append([int(x2), int(y2)])

    if show :
        # Draw top 4 matches
        result = cv2.drawMatches(cur_img_uint8, kp1, des_img_uint8, kp2, 
                                matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(15, 6), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        plt.imshow(result)
        plt.title(f'Top 4 SIFT Feature Matches', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()  

    return cur_features, des_features    
    



def get_feats_depth(features, depth_map) :
    #the depth map got the cam_img_H, img_W, and the z value
    #the features are list of u,v (pxls coord)
    features_depth = []
    for f  in features :
        if(depth_map[f[1],f[0]]>100) :
            depth_map[f[1],f[0]] = 100

        if(depth_map[f[1],f[0]] < 0) :
            depth_map[f[1],f[0]] = 0.5
            print("some features with z < 0?")
        
        if(depth_map[f[1],f[0]] == 0) :
            depth_map[f[1],f[0]] = 0.5
        
        features_depth.append(depth_map[f[1],f[0]])
    return np.array(features_depth)



# -------------- Get errors --------------------------------
def getting_errors(Ss : List[List[float]], Ss_star : List[List[float]]) -> List[float]:  
    
    return [a - b for row1, row2 in zip(Ss, Ss_star) for a, b in zip(row1, row2)]



# ------------- Getting L and pseudo L --------------------
def get_interaction_matrix(nbr_features, Ss, Ss_z, f):

    Ls = np.zeros((nbr_features, 2, 6), dtype=float)

    for i, s in enumerate(Ss) :
        x=s[0]
        y=s[1]
        Z=Ss_z[i]

        # Build interaction matrix (2x6)
        L = np.zeros((2, 6))
        L[0, 0] = -f / Z
        L[0, 1] = 0.0
        L[0, 2] = x / Z
        L[0, 3] = x * y / f
        L[0, 4] = -(f + (x * x) / f)
        L[0, 5] = y
        L[1, 0] = 0.0
        L[1, 1] = -f / Z
        L[1, 2] = y / Z
        L[1, 3] = f + (y * y) / f
        L[1, 4] = -(x * y) / f
        L[1, 5] = -x

        Ls[i] = L

    return Ls


def get_inter_mat_pseudo_inverse(intr_mat):

    # make sure the intr matrix got N,2,6 shape 
    intr_mat = np.asarray(intr_mat, dtype=float)
    if intr_mat.ndim != 3 or intr_mat.shape[1:] != (2, 6):
        raise ValueError(f"intr_mat must have shape (N,2,6), got {intr_mat.shape}")

    # Flatten to 2D (2N x 6), so we can calc pseudo inv
    N = intr_mat.shape[0]
    L_stack = intr_mat.reshape(2*N, 6)

    # Compute Moore-Penrose pseudo-inverse
    pseudo_intr_mat = np.linalg.pinv(L_stack)  # shape (6 x 2N)

    return pseudo_intr_mat





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




def save_img(img, title, folder_path) :
    img = np.asarray(img)
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{folder_path}/{title}.png", img_u8)




def initialize_traject_visualizer(vis, trajectory_line, des_cam_axis, cur_cam_axis, cur_extrins, mesh) :

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
    T_cw = get_homog([1, -1.8, -1.5, 0, 0, 0])
    T_cw1 = get_homog([-np.pi/2, 0, 0, 0, 0, 0])
    T_cw2 = get_homog([0, 0, 0, 0, 0, 0])
    pov_extrins = cur_extrins @ T_cw @ T_cw1 @ T_cw2
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
    cur_camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
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

    return prev_camera_axis, np.asarray(img)




def plot_points_on_image(image, points, point_size=30):
    """
    image  : input image (H, W, 3), RGB
    points : list or array of [u, v] pixel coordinates
    point_size : size of plotted points
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    if len(points) > 0:
        u = [p[0] for p in points]
        v = [p[1] for p in points]
        plt.scatter(u, v, c='red', s=point_size)

    plt.axis("off")
    plt.tight_layout()
    plt.show()





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














# -------------------------
# Main Pipeline
# -------------------------

def main():

    # Fixed Vars :
    lambda_gain = 0.5
    dt = 0.1
    nbr_features = 10


    # Setup init nd desired homog matrices nd axises
    cur_pose_vect = [2.2, 0, -2, 0, 0, 0] 
    cur_extrins, des_extrins, des_cam_axis, cur_cam_axis = set_cameras(cur_pose_vect)
    #oscill : cur_pose_vect = [3, -0.3, -2, 0, 0, 0]  
    #works : cur_pose_vect = [2.2, 0, -2, 0, 0, 0] 


    # Get init 3d points from img with moge (the o3d to render, nd cam nd world coords))
    points_o3d, points_world, points_cam = load_points(img_path, cur_extrins)


    # Rendering from init cam nd desired 
    des_img, des_depth_map = takin_pic(points_o3d, des_extrins)
    gray_des = 0.299 * des_img[:, :, 0] + 0.587 * des_img[:, :, 1] + 0.114 * des_img[:, :, 2] #just for diff visualization
    cur_img, cur_depth_map = takin_pic(points_o3d, cur_extrins)
    save_img(des_img, "des", "frames")
    plot_img(des_img, "des")


    # Getting top n matched features
    des_kp, des_desc = get_sift_features(des_img)
    cur_kp, cur_des = get_sift_features(cur_img)
    cur_feats, des_feats = get_matches(nbr_features, cur_kp, cur_des, des_kp, des_desc, cur_img, des_img, show= True)


    # Get 3d points in wf corresponding to the features using (curfeats nd curimgs)   
    fixed_points_wf = []
    for j in range(len(cur_feats)) :
        fixed_points_wf.append(points_world[cur_feats[j][1], cur_feats[j][0]])
        # Points_world (H, W, 3(xyz)) in world frame corresponding to original init image, and cur_feats is a list of [u,v] (of ftrs) from rendered image


    # Get the corresponding Ss_star of these 3d points
    fixed_points_cf = LinAlgeb.transform_inverse(fixed_points_wf, des_extrins)
    Ss_star = get_Ss_from_points(fixed_points_cf)


    # intialize trajectory scene : the visualizer with frames, mesh and line
    camera_centers = []
    trajectory_line = o3d.geometry.LineSet()
    o3d_vis = o3d.visualization.Visualizer()
    initialize_traject_visualizer(o3d_vis, trajectory_line, des_cam_axis, cur_cam_axis, cur_extrins, points_o3d)
    prev_cur_frame = None



    try:    
        for i in range(500) :

            # 0 - Update visualizer of trajects
            prev_cur_frame, scene_img = update_traject_visualizer(i, o3d_vis, trajectory_line, camera_centers, cur_extrins, prev_cur_frame)

            # 1 - Rendering current img , nd get depth exprssd camera frame with cam frame units 
            cur_img, cur_depth_map = takin_pic(points_o3d, cur_extrins)
            save_img(cur_img, i, "frames")

            # make our matplotlib vis
            if(i==0) :
                matp_vis = LiveOptimizationVisualizer(des_img, cur_img)

            # 2 - Get corresponding Ss of these fixed 3d points
            fixed_points_cf = LinAlgeb.transform_inverse(fixed_points_wf, cur_extrins)
            Ss = get_Ss_from_points(fixed_points_cf)
            
            # 3 - Get corresp uvs and their depth in meters
            cur_feats_uv = get_uv_from_Ss(Ss)
            Ss_Z = get_feats_depth(cur_feats_uv, cur_depth_map)

            # 4 - Getting list of errors nd reshaping it
            errors = getting_errors(Ss, Ss_star) 
            errors = np.asarray(errors, dtype=float).reshape(-1)

            # 5 - Print the norm of the error and get the diff of imgs
            norm_of_error = np.linalg.norm(errors) 
            gray_cur = 0.299 * cur_img[:, :, 0] + 0.587 * cur_img[:, :, 1] + 0.114 * cur_img[:, :, 2]
            
            current_diff_img = compute_grayscale_difference(gray_cur, gray_des)
            print(f"error {i} :", norm_of_error)

            if norm_of_error < 0.01 :
                break

            # 6 - Get the intr matrix nd its pseudo_inv 
            L = get_interaction_matrix(nbr_features, Ss, Ss_Z, 1)
            L_psinv = get_inter_mat_pseudo_inverse(L)

            # 7 - Control law after
            V = - lambda_gain * (L_psinv @ errors)    

            # 8 - Apply the velocity for dt and update cam pose
            cur_extrins = update_cam_pose(cur_extrins, V, dt)

            matp_vis.update(i, scene_img, cur_img, current_diff_img, V, norm_of_error)
    
        matp_vis.close()
        o3d_vis.close()

    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            matp_vis.close()
            o3d_vis.close()
            os._exit(0)

    



if __name__ == "__main__":
    main()