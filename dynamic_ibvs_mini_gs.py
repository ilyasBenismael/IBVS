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





#_________IBVS Vars :
lambda_gain = 0.01
dt = 0.1
nbr_features = 10




#________our robot camera
CAM_W, CAM_H = 1500, 1000
FX = FY = 0.8 * max(CAM_W, CAM_H)
f=1
CX, CY = CAM_W / 2.0, CAM_H / 2.0
#intrinsic format for o3d camera
intrins = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W,
    height=CAM_H,
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY
)
#intrinsic format for gsplat rendering
Ks = torch.tensor(
    [[FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],],
    dtype=torch.float32,
    device="cuda",
).unsqueeze(0)


#_____trajectory visualier camera 
CAM_W2, CAM_H2 = 400, 300
FX2 = FY2 = 0.8 * max(CAM_W2, CAM_H2)
CX2, CY2 = CAM_W2 / 2.0, CAM_H2 / 2.0
#o3d format
intrins2 = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W2,
    height=CAM_H2,
    fx=FX2,
    fy=FY2,
    cx=CX2,
    cy=CY2
)



#___________Paths : 
gs_path = "gs_scenes/office2_10_250.ply"
mesh_path = "meshes/office2.glb"
sfm_recons_path = "sfm_scenes/office2_10imgs/sparse/0"
sfm_images_path = "sfm_scenes/office2_10imgs/images"
gs_frames_path = "frames/minigs"



#__________________________________________________________________________________________________





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




def set_init_cam(mesh, save = False) :

    # making athe world axis // default args keep position 000 matching wf nd also the size as 1 same as wf units
    init_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

    # Making the des_pose centered in front of the scene by transforming wf with 2 succssf trnsfs
    T_cw1 = get_homog([0, 0, 0, 0, 0, -np.pi])  
    T_cw2 = get_homog([0, 0, 0, 0, np.pi/2, 0])  
    T_cw3 = get_homog([0, 0, -10, 0, 0, 0])
    init_extrins = T_cw1 @ T_cw2 @ T_cw3
    init_cam_axis.transform(init_extrins)
    image, _ = takin_pic(mesh, init_extrins)

    if(save) :
        save_img(image, 1, sfm_images_path)
    
    return init_extrins, init_cam_axis, image





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
        cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
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
    cv2.imwrite(f"{gs_frames_path}/{i}.png", img)
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





def initialize_traject_visualizer(vis, trajectory_line, des_cam_axis, cur_cam_axis, init_extrins, mesh) :

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
    T_cw = get_homog([0, 0, -3, 0, 0, 0])
    T_cw1 = get_homog([0, 0, 0, 0, 0, 0])
    T_cw2 = get_homog([0, 0, 0, 0, 0, 0])
    pov_extrins = init_extrins @ T_cw @ T_cw1 @ T_cw2
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





def set_init_cams(
    mesh,
    init_o3d_cam_pose,
    motions,
    axis_size=1.5,
    save = False
):

    o3d_cam_poses = []
    o3d_cam_axises = []
    images = []

    for i, m in enumerate(motions, start=2):

        # build transform
        T = (
            init_o3d_cam_pose
            @ get_homog([*m["t"], 0, 0, 0])
            @ get_homog([0, 0, 0, *m["r"]])
        )

        img, _ = takin_pic(mesh, T)
        if(save) :
            save_img(img, i, "frames/sfm/images")
        images.append(img)
        o3d_cam_poses.append(T)


        # create axis
        cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size
        )

        cam_axis.transform(T)

        o3d_cam_axises.append(cam_axis)


    return o3d_cam_poses, o3d_cam_axises, images

    





#__________________________________________________________________________________


def main() :


    #_________IBVS Vars :
    lambda_gain = 0.01
    dt = 0.1
    nbr_features = 10


    # I)__________________ setting up real world (o3d) :

    # o3d images, the images that we will be using for the whole pipeline, numpy hw3 each
    all_o3d_images = []

    # load the o3d mesh  
    scene_mesh = load_mesh(mesh_path, lambert=False)

    # Defining our initial camera we set it manually inside the function
    init_o3d_cam_pose, init_o3d_cam_axis, init_o3d_img = set_init_cam(scene_mesh, save = False)
    all_o3d_images.append(init_o3d_img)

    
    # Manually choosing the number of initial cams poses (relatively to intial o3d camera) :
    # choosing 10 init randome vectors (relative to init)
    # these are the same o3d poses I used for sfm
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
    
    # Getting the 10 init cam poses and axises (i will make it save the images for sfm if it's the first time)
    init_o3d_cams_poses, init_o3d_cams_axises, images = set_init_cams(scene_mesh, init_o3d_cam_pose, motions, save=False)
    all_o3d_images.append(images)
    compos = [scene_mesh, init_o3d_cam_axis, *init_o3d_cams_axises]
    visualize_scene(compos)

    # getting desired o3d axis, i wanna get 
    init_o3d_cam_pose, init_o3d_cam_axis = init_o3d_cams_poses[3], init_o3d_cams_axises[3]
    des_o3d_cam_axis = init_o3d_cams_axises[8]



    # --> SFM is done in this step before we continue


    

    # II)_____________Loading the gs-sfm scene : 
    
    # Loading gaussians from the ply :
    means, quats, scales, opacities, sh = load_gaussians_from_ply(gs_path)

    

    # Loading the sfm-cameras (in w->c) and 3d points
    recon = pycolmap.Reconstruction(sfm_recons_path)

    # Loading the sfm cams will load the same ones above with the same names of images and their relative poses in sfm units
    init_sfm_poses, init_sfm_axises = get_sfm_poses(recon)
    print(init_sfm_poses)
    xyz, colors, sfm_o3d_points = get_3dpoints(recon)
    z_mean = xyz[:, 2].mean()
    print("xyz shape", xyz.shape)
    print("colors shape", colors.shape)
    print("Average Z:", z_mean)
    print("Number of points:", xyz.shape[0])

    compos = [*init_sfm_axises, sfm_o3d_points]
    visualize_scene(compos)

    
    # choosing initial image 
    cur_T = init_sfm_poses["8.png"]# as the first one in o3d which corresponds in sfm poses the 7th one
    # keeping cur_T in cf
    cur_T = np.linalg.inv(cur_T)
    #getting the depth map as avrg z of point
    depth_map = np.full((CAM_W, CAM_H), z_mean)
        
    # 3- rendering from a posed-cam as desired pose using gsplat (we get des_T in wf)
    des_T = init_sfm_poses["9.png"]
    des_T = np.linalg.inv(des_T)
    #des_T = cur_T @ get_homog([-1, -0.5, 0, 0, 0, 0])

    # turn it to wf then to torch
    des_viewmat = make_viewmat(des_T)
    des_img = render_view(means, quats, scales, opacities, sh, des_viewmat, Ks, CAM_W, CAM_H)
    gray_des = 0.299 * des_img[:, :, 0] + 0.587 * des_img[:, :, 1] + 0.114 * des_img[:, :, 2]
    save_img(des_img, f"desired_img", gs_frames_path)

    # Get keypoints from desired img
    des_kp, des_desc = get_features(des_img)

    
    # Intialize trajectory scene (sfm) : the visualizer with sfm cams and sfm 3d points
    camera_centers_sfm = []
    trajectory_line_sfm = o3d.geometry.LineSet()
    sfm_vis = o3d.visualization.Visualizer()
    des_cam_axis_sfm = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    des_cam_axis_sfm.transform(des_T)
    cur_cam_axis_sfm = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cur_cam_axis_sfm.transform(cur_T)
    initialize_traject_visualizer(sfm_vis, trajectory_line_sfm, des_cam_axis_sfm, cur_cam_axis_sfm, cur_T, sfm_o3d_points)
    prev_sfm_axis = None
    

    # Intialize trajectory scene (o3D) : the visualizer with o3d cams and o3d mesh 
    camera_centers_o3d = []
    trajectory_line_o3d = o3d.geometry.LineSet()
    o3d_vis = o3d.visualization.Visualizer()    
    initialize_traject_visualizer(o3d_vis, trajectory_line_o3d, init_o3d_cam_axis, init_o3d_cam_axis, init_o3d_cam_pose, scene_mesh)
    prev_o3d_axis = None



    try:
        for i in range(999):

            prev_sfm_axis, scene_img_sfm = update_traject_visualizer(i, sfm_vis, trajectory_line_sfm, camera_centers_sfm, cur_T, prev_sfm_axis)
            prev_o3d_axis, scene_img_o3d = update_traject_visualizer(i, o3d_vis, trajectory_line_o3d, camera_centers_o3d, init_o3d_cam_pose, prev_o3d_axis)

            # 1 - taking pic from cur_T (this function turn it to cf then torch for gsplat to use)
            cur_viewmat = make_viewmat(cur_T) 
            cur_img = render_view(means, quats, scales, opacities, sh, cur_viewmat, Ks, CAM_W, CAM_H)
            gray_cur = 0.299 * cur_img[:, :, 0] + 0.587 * cur_img[:, :, 1] + 0.114 * cur_img[:, :, 2]
            

            # getting cur features
            cur_kp, cur_des = get_features(cur_img)
            
            show =False
            if(i==0):
                #making our matp visualizer
                matp_vis = LiveOptimizationVisualizer(des_img, cur_img)
                save_img(cur_img, i, "frames")
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
            init_o3d_cam_pose = update_cam_pose(init_o3d_cam_pose, V, dt)
            matp_vis.update(i, match_img, cur_img, current_diff_img, V, norm_of_error)


        matp_vis.close()
        o3d_vis.close()
        


    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting loop cleanly.")
        matp_vis.close()
        o3d_vis.close()





if __name__ == "__main__":
    main()