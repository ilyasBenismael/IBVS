import pycolmap
import torch
import numpy as np
from plyfile import PlyData
from utils.lin_algeb import LinAlgeb
import matplotlib.pyplot as plt
from gsplat import rasterization
from typing import Tuple, Sequence, List, Union
import cv2
import math
from datetime import datetime
from utils.visualizer import LiveOptimizationVisualizer
import open3d as o3d








# Cam vars

CAM_W, CAM_H = 1500, 1000
FX = FY = 0.8 * max(CAM_W, CAM_H)
f=1
CX, CY = CAM_W / 2.0, CAM_H / 2.0
Ks = torch.tensor(
    [[FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],],
    dtype=torch.float32,
    device="cuda",
).unsqueeze(0)



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



# IBVS Vars :
lambda_gain = 0.1
dt = 0.1
nbr_features = 10


#Paths : 
path = "scenes/office_250.ply"











def get_homog(pose_vector) :
    R,t = LinAlgeb.make_rot_trans(*pose_vector)
    return LinAlgeb.get_homog_matrix(R,t)





def load_gaussians_from_ply(path, device="cuda"):

    # Read PLY
    ply = PlyData.read(path)["vertex"]
    N = ply.count
    print(f"Loaded {N} vertices from {path}\n")

    # -----------------------
    # Means
    # -----------------------
    means = torch.stack([
        torch.from_numpy(ply["x"]),
        torch.from_numpy(ply["y"]),
        torch.from_numpy(ply["z"]),
    ], dim=1).float().to(device)

    # -----------------------
    # Scales (log → real)
    # -----------------------
    scales_log = torch.stack([
        torch.from_numpy(ply["scale_0"]),
        torch.from_numpy(ply["scale_1"]),
        torch.from_numpy(ply["scale_2"]),
    ], dim=1).float().to(device)
    scales = torch.exp(scales_log)

    # -----------------------
    # Rotation (normalize quaternion)
    # -----------------------
    quats = torch.stack([
        torch.from_numpy(ply["rot_0"]),
        torch.from_numpy(ply["rot_1"]),
        torch.from_numpy(ply["rot_2"]),
        torch.from_numpy(ply["rot_3"]),
    ], dim=1).float().to(device)
    quats = quats / torch.norm(quats, dim=1, keepdim=True)


    # -----------------------
    # Opacity (inverse sigmoid → alpha)
    # -----------------------
    opacity_param = torch.from_numpy(ply["opacity"]).float().to(device)
    opacities = torch.sigmoid(opacity_param)

    # -----------------------
    # Spherical Harmonics
    # -----------------------
    f_dc = torch.stack([
        torch.from_numpy(ply["f_dc_0"]),
        torch.from_numpy(ply["f_dc_1"]),
        torch.from_numpy(ply["f_dc_2"]),
    ], dim=1)

    # Compute number of rest coefficients
    ply_data = ply.data  # <-- access the structured array
    rest_keys = [k for k in ply_data.dtype.names if k.startswith("f_rest_")]

    f_rest = torch.stack([
        torch.from_numpy(ply_data[k]) for k in rest_keys
    ], dim=1)


    sh = torch.cat([f_dc, f_rest], dim=1)   # (N, num_coeffs*3)
    sh = sh.view(N, -1, 3).float().to(device)




    return means, quats, scales, opacities, sh





def make_viewmat(T_wc, device="cuda"):
    viewmat = torch.tensor(T_wc, dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(viewmat)
    return viewmat.unsqueeze(0)




def render_view(means, quats, scales, opacities, sh, T, K, W, H):

    image, alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh,
        viewmats=T,
        Ks=K,
        width=W,
        height=H,
        sh_degree=2,              
        rasterize_mode="antialiased",
        render_mode="RGB",
    )

    # we get the only img in the batch with image[0], we move the tensor to cpu nd turn it to numpy
    img = image[0].detach().cpu().numpy()

    return img




def plot_img(img, title) :
    plt.imshow(img)
    plt.axis("off")
    plt.suptitle(title)
    plt.show()






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




def visualize_scene(scene_compos) :
        o3d.visualization.draw_geometries(
        scene_compos,
        window_name="The scene",
        width=800,
        height=800,
        mesh_show_back_face=True)



def get_sfm_poses(recon) : 
    axises = []
    homog_poses = {}
    for image in recon.images.values():
        if not image.has_pose:
            continue
        T_cw = image.cam_from_world()   # Rigid3d
        R = T_cw.rotation.matrix()      # (3,3)
        t = T_cw.translation            # (3,)
        T_h = LinAlgeb.get_homog_matrix(R,t)
        homog_poses[image.name] = T_h

        # Making the axises
        # turn to cam->w for o3d
        T_vis = np.linalg.inv(T_h)
        cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        cam_axis.transform(T_vis)
        axises.append(cam_axis)
        
    return homog_poses, axises




def get_3dpoints(recon) :
    colors = []
    xyz_list = []

    for p3d in recon.points3D.values():
        colors.append(p3d.color)
        xyz_list.append(p3d.xyz)

    xyz_list = np.array(xyz_list)
    colors = np.array(colors) # colors are 0-1

    # Create o3d_point_cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_list)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return xyz_list, colors, pcd





def get_features(img):

    # Convert normalized RGB to uint8 format
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Convert RGB to grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des







def get_matches(nbr_features, kp1, des1, kp2, des2, cur_img, des_img, show, i) :

    cur_img_uint8 = (cur_img * 255).astype(np.uint8)
    des_img_uint8 = (des_img * 255).astype(np.uint8)
    
    # Match features using BFMatcher with L2 norm (for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (best matches first) and keep top 4
    matches = sorted(matches, key=lambda x: x.distance)[:nbr_features]

    if len(matches) < nbr_features :
        print(f"got just {len(matches)} matches")
        nbr_features = len(matches)
        if(len(matches)) < 3 : 
            ValueError(f"less than 3 features : {len(matches)}")
    
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

    
    # Draw top 4 matches
    result = cv2.drawMatches(cur_img_uint8, kp1, des_img_uint8, kp2, 
                                matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)    

    img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    cv2.imwrite(f"frames/fibvs_4matches/{i}.png", img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    

    if show :
        plt.figure(figsize=(15, 6), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        plt.imshow(result)
        plt.title(f'Top 4 SIFT Feature Matches', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()  

    return cur_features, des_features, img, nbr_features   





def get_Ss_from_uv(uv_list):
    Ss = []
    for u, v in uv_list:
        x = (u - CX) / FX
        y = (v - CY) / FY
        Ss.append([x, y])
    return np.array(Ss)






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




    


#__________________________________________________________________________________


def main() :

    # 1 - loading gaussians from the ply :
    means, quats, scales, opacities, sh = load_gaussians_from_ply(path)
    nbr_features = 10


    # 2-loading the sfm-cameras and 3d points
    recon = pycolmap.Reconstruction("/home/user/Bureau/visual_navigation/databases/mini_office_sfm/sparse/0")
    poses_imgs, axises = get_sfm_poses(recon)
    xyz, colors, o3dpoints = get_3dpoints(recon)
    z_mean = xyz[:, 2].mean()
    print("xyz shape", xyz.shape)
    print("colors shape", colors.shape)
    print("Average Z:", z_mean)
    print("Number of points:", xyz.shape[0])

    compos = [*axises, o3dpoints]
    visualize_scene(compos)


    # getting init (cur_T) pose
    _, cur_T = list(poses_imgs.items())[5]
    # keeping cur_T in cf
    cur_T = np.linalg.inv(cur_T)
    #getting the depth map as avrg z of point
    depth_map = np.full((CAM_W, CAM_H), z_mean)
        
    # 3- rendering from a posed-cam as desired pose using gsplat (we get des_T in wf)
    des_T = cur_T @ get_homog([0.7, 0.2, 0.1, np.pi/25, 0, 0])


    # turn it to wf then to torch
    des_viewmat = make_viewmat(des_T)
    des_img = render_view(means, quats, scales, opacities, sh, des_viewmat, Ks, CAM_W, CAM_H)
    gray_des = 0.299 * des_img[:, :, 0] + 0.587 * des_img[:, :, 1] + 0.114 * des_img[:, :, 2]
    save_img(des_img, f"desired_img", "frames/fibvs_4matches")

    # Get keypoints from desired img
    des_kp, des_desc = get_features(des_img)


     # Intialize trajectory scene : the visualizer with frames, mesh and line
    camera_centers = []
    trajectory_line = o3d.geometry.LineSet()
    o3d_vis = o3d.visualization.Visualizer()
    des_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    cur_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    initialize_traject_visualizer(o3d_vis, trajectory_line, des_cam_axis, cur_cam_axis, des_T, o3dpoints)
    prev_cur_frame = None





    try:
        for i in range(999):


            prev_cur_frame, scene_img = update_traject_visualizer(i, o3d_vis, trajectory_line, camera_centers, cur_T, prev_cur_frame)

            # 1 - taking pic from cur_T (this function turn it to cf then torch for gsplat to use)
            cur_viewmat = make_viewmat(cur_T) 
            cur_img = render_view(means, quats, scales, opacities, sh, cur_viewmat, Ks, CAM_W, CAM_H)
            gray_cur = 0.299 * cur_img[:, :, 0] + 0.587 * cur_img[:, :, 1] + 0.114 * cur_img[:, :, 2]
            #save_img(cur_img, i, "frames")

            # getting cur features
            cur_kp, cur_des = get_features(cur_img)
            
            show =False
            if(i==0):
                #making our matp visualizer
                matp_vis = LiveOptimizationVisualizer(des_img, cur_img)
                show = True

            cur_feats, des_feats, match_img, nbr_features = get_matches(nbr_features, cur_kp, cur_des, des_kp, des_desc, cur_img, des_img, show, i)
            
            # Get Ss Ss* and Ss_depths
            Ss = get_Ss_from_uv(cur_feats)
            Ss_star = get_Ss_from_uv(des_feats)
            Ss_Z = np.full((nbr_features), z_mean)

            # Getting list of errors nd reshaping it
            errors = getting_errors(Ss, Ss_star) 
            errors = np.asarray(errors, dtype=float).reshape(-1)
            current_diff_img = compute_grayscale_difference(gray_cur, gray_des)


            # Print the norm of the error
            norm_of_error = np.linalg.norm(errors) 
            print(f"error {i} :", norm_of_error)

            if norm_of_error < 0.001 :
                break

            # Get the intr matrix nd its pseudo_inv 
            L = get_interaction_matrix(nbr_features, Ss, Ss_Z, 1)
            L_psinv = get_inter_mat_pseudo_inverse(L)

            # Control law after
            V = - lambda_gain * (L_psinv @ errors)    

            # Apply the velocity for dt and update cam pose
            cur_T = update_cam_pose(cur_T, V, dt)
            matp_vis.update(i, scene_img, cur_img, current_diff_img, V, norm_of_error)


        matp_vis.close()
        o3d_vis.close()
        


    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting loop cleanly.")
        matp_vis.close()
        o3d_vis.close()





if __name__ == "__main__":
    main()