import os
import sys
import open3d as o3d
import numpy as np
from utils.lin_algeb import LinAlgeb
import cv2
import matplotlib.pyplot as plt
import math
from plyfile import PlyData, PlyElement
from PIL import Image
from datetime import datetime
from utils.visualizer import LiveOptimizationVisualizer
import torch
from typing import Tuple, Sequence, List, Union
from moge.model.v2 import MoGeModel
from gsplat import rasterization
import utils3d 

# Add accelerated_features to our Python paths , so that when featx script gets executed it xill know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat 






#ibvs params
dt = 0.4
lamda = 0.1

#paths
mesh_path = "meshes/office2.glb"
o3d_frames_path = "frames/o3d"
gs_frames_path = "frames/gs"
moge_points_save_path = "moge_points/office2_0.ply"
gs_save_path = "gs_scenes/init_gs_office2.ply"


# robot camera
CAM_W, CAM_H = 1200, 1000
FX = FY = 0.8 * max(CAM_W, CAM_H)
f=1
CX, CY = CAM_W / 2.0, CAM_H / 2.0
intrins_o3d = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W,
    height=CAM_H,
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY
)

intrins_gs = torch.tensor(
    [[FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],],
    dtype=torch.float32,
    device="cuda",
).unsqueeze(0)

# whole scene visualizer camera
CAM_W2, CAM_H2 = 400, 300
FX2 = FY2 = 0.8 * max(CAM_W2, CAM_H2)
CX2, CY2 = CAM_W2 / 2.0, CAM_H2 / 2.0
intrins2_o3d = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W2,
    height=CAM_H2,
    fx=FX2,
    fy=FY2,
    cx=CX2,
    cy=CY2
)

np.set_printoptions(precision=2, suppress=False)










def load_np_img(img_path) :
    img = Image.open(img_path).convert("RGB")
    return np.array(img)


def visualize_scene(scene_compos) :
        o3d.visualization.draw_geometries(
        scene_compos,
        window_name="The scene",
        width=800,
        height=800,
        mesh_show_back_face=True)



def get_cam_pose_from_mesh_view(mesh):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Choose ur pose", width=800, height=800)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    print("choose ur pose and close window")
    
    vis.run()
    
    # Get camera parameters
    vc = vis.get_view_control()
    cam_params = vc.convert_to_pinhole_camera_parameters()
    
    # Extract pose turn if frm wrld relative to cam into cam relative to world 
    T_wc = cam_params.extrinsic  
    T_cw = np.linalg.inv(T_wc)

    vis.destroy_window()
    return T_cw




def get_homog(pose_vector) :
    R,t = LinAlgeb.make_rot_trans(*pose_vector)
    return LinAlgeb.get_homog_matrix(R,t)





def load_mesh(path, scale, lambert) :
    
    # loading the mesh, enable post to render the colored texture, (no normals for lambertian)
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    print(f"Number of vertices: {len(mesh.vertices)}")

    if(not lambert) :
        mesh.compute_vertex_normals()  

    # center the mesh, and scaling it
    mesh.translate(-mesh.get_center()) 
    mesh.scale(scale , center=mesh.get_center()) #this center is just about mesh position after scaling (keepin it in the center here)

    return mesh 





def get_moge_points(img, threshold=0.01):
    
    # Loading MoGe
    device = "cuda"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()

    # Making sure img is in 0-1
    if img.max() > 1.0:
        img = img / 255.0

    # turn img to torch and apply moge
    image = torch.from_numpy(img).float().to(device).permute(2, 0, 1)
    output = model.infer(image)


    points = output["points"].cpu().numpy()  # (H, W, 3)
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()

    #check the edges (big depth diffs) and add it to the mask area to remove
    edge_mask = utils3d.np.depth_map_edge(depth, rtol=threshold)
    mask_cleaned = mask & (~edge_mask)

   
    
    # get colors frm img and flatten all (colors, points, masks)
    colors = img.astype(np.float64)
    colors_flat = colors.reshape(-1, 3)
    pts_flat = points.reshape(-1, 3).astype(np.float64)
    valid_flat = mask_cleaned.reshape(-1)

    # save clean points and colors
    final_points = pts_flat[valid_flat]
    final_colors = colors_flat[valid_flat]

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(final_points)
    o3d_points.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(moge_points_save_path, o3d_points)

    # Return FULL arrays (H, W, 3) for indexing by pixel coordinates
    return o3d_points, final_points, final_colors  






    
def render_mesh_pic(mesh, extrins):

    # defining extrins and T_cw_final params 
    extrins = np.linalg.inv(extrins)

    # making our pincamparams objct
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = intrins_o3d
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




def plot_2_imgs(img1, img2, title1="", title2="", suptitle=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img1)
    axs[0].axis("off")
    axs[0].set_title(title1)

    axs[1].imshow(img2)
    axs[1].axis("off")
    axs[1].set_title(title2)

    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()





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





def get_uv_from_Ss(Ss):
    uv_list = []
    for x, y in Ss:
        u = CX + FX * x 
        v = CY + FY * y
        u = int(np.clip(round(u), 0, CAM_W - 1))
        v = int(np.clip(round(v), 0, CAM_H - 1))
        uv_list.append([u, v])
    return np.array(uv_list)






def save_img(img, title, folder_path, assume_rgb=True):
    os.makedirs(folder_path, exist_ok=True)
    img = np.asarray(img, dtype=np.float32)
    img_u8 = (img * 255 if img.max() <= 1.0 else img).clip(0, 255).astype(np.uint8)
    if assume_rgb:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(folder_path, f"{title}.png"), img_u8)



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
        plt.title(f'SIFT Feature Matches', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()  

    return cur_features, des_features    
    





def get_feats_depth(features, depth_map) :
    #the depth map got the cam_img_H, img_W, and the z value
    #the features are list of u,v (pxls coord)
    features_depth = []
    for f  in features :
        
        if(depth_map[f[1],f[0]]>1000) :
            depth_map[f[1],f[0]] = 1000
            print("some features with z > 1000")

        if(depth_map[f[1],f[0]] < 0) :
            print("some features with z < 0")
        
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



"""
def filter_valid_points(points_list, min_points=3):
    valid = [np.array(p) for p in points_list 
             if np.isfinite(p).all() and (len(p) < 3 or p[2] > 0)]
    
    if len(valid) < min_points:
        raise ValueError(f"Only {len(valid)}/{len(points_list)} valid points (need {min_points})")
    
    return valid

"""


def filter_valid_points(candidate_points, cur_pose, des_pose, nbr):

    points_in_cur_cam = LinAlgeb.transform_points_to_cam(candidate_points, cur_pose)
    points_in_des_cam = LinAlgeb.transform_points_to_cam(candidate_points, des_pose)

    # Positive depth in both cameras
    valid_mask = (points_in_cur_cam[:, 2] > 0) & (points_in_des_cam[:, 2] > 0)

    valid_indices = np.where(valid_mask)[0]

    # check if enough points are detected
    if len(valid_indices) < nbr:
        raise ValueError(
            f"Only {len(valid_indices)} valid points detected, "
            f"but {nbr} required."
        )

    selected_indices = valid_indices[:nbr]

    return candidate_points[selected_indices]







def get_Ss_from_uv(uv_list):
    Ss = []
    for u, v in uv_list:
        x = (u - CX) / FX
        y = (v - CY) / FY
        Ss.append([x, y])
    return np.array(Ss)



""""

def draw_matches(cur_img, des_img, pts1, pts2):

    # making sure the matches are equal
    assert len(pts1) == len(pts2), "Point lists must have same length"
    n = len(pts1)

    cur_img = cur_img.copy()
    des_img = des_img.copy()

    # Ensure uint8 (format waited by open_cv [0-255])
    if cur_img.dtype != np.uint8:
        cur_img = (cur_img * 255).astype(np.uint8)
    if des_img.dtype != np.uint8:
        des_img = (des_img * 255).astype(np.uint8)

    # Generate n random colors
    colors = (np.random.rand(n, 3) * 255).astype(np.uint8)

    #for each elemnt of the uvs : get a color frm colors and draw it in the uv elmnt on both imgs
    for k, ((u1, v1), (u2, v2)) in enumerate(zip(pts1, pts2)):
        r, g, b = colors[k]
        color_bgr = (int(b), int(g), int(r))
        # filled colored circles on uv
        cv2.circle(cur_img, (int(u1), int(v1)), 6, color_bgr, -1)
        cv2.circle(des_img, (int(u2), int(v2)), 6, color_bgr, -1)

    return cur_img, des_img

"""





def init_gaussians_from_points(
    xyz,
    rgb,
    ply_path,
    scale=0.005,
    alpha=0.9,
    sh_degree=2,
    device="cuda",
):


    # Flatten inputs
    xyz = xyz.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    N = xyz.shape[0]

    # Normalize moge_colors
    if rgb.max() > 1.0:
        rgb = rgb / 255.0


    # Get f_dc from rgb
    f_dc = (rgb - 0.5) / 0.28209479177387814

    # Defining the nmbr of sh coeffs
    num_sh = (sh_degree + 1) ** 2

    #______ init opacity
    # Ok in gs we have the opacity the value we optimize and which we use inside a sigmoid to turn to 0-1, gsplat use the 0-1 one (for render), ply idk wht it stores
    #as i see that gsplate takes the sigmoid function, the one i should give as input, and keep it for  
    opacity_logit = LinAlgeb.inverse_sigmoid(alpha)

    #______ init opacity
    # when optimizing the scale we do exp(scale) so it is always positive, but ply wants the log value (optimized one) and the gsplat want the exp one (render one)
    scale_log = np.log(scale)


    # ______PLY file structure (GS-compatible)
    
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),          # Gaussian center
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),       # Unused normals
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),  # SH DC
        ("opacity", "f4"),                              # Opacity logit
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),  # log-scales
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),  # quaternion
    ]

    # Higher-order SH coefficients set to 0
    for i in range(num_sh - 1):
        for c in range(3):
            dtype.append((f"f_rest_{3*i + c}", "f4"))

    data = np.empty(N, dtype=dtype)



    #____________  Fill PLY data

    # Positions
    data["x"], data["y"], data["z"] = xyz.T

    # Normals are unused by GS → set to zero
    data["nx"] = data["ny"] = data["nz"] = 0.0

    # SH DC coefficients
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]

    # Opacity
    data["opacity"] = opacity_logit

    # Scales
    data["scale_0"] = scale_log
    data["scale_1"] = scale_log
    data["scale_2"] = scale_log

    # Identity rotation quaternion
    data["rot_0"] = 1.0
    data["rot_1"] = 0.0
    data["rot_2"] = 0.0
    data["rot_3"] = 0.0

    # Higher-order SH coefficients start at zero
    for i in range(num_sh - 1):
        for c in range(3):
            data[f"f_rest_{3*i + c}"] = 0.0

    # Save ply file
    PlyData([PlyElement.describe(data, "vertex")]).write(ply_path)


    # Turn these gaussians properties to tensors
    means = torch.from_numpy(xyz).float().to(device)
    scales = torch.full((N, 3), scale, device=device)
    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0  # identity rotation
    opacities = torch.full((N,), alpha, device=device)

    # SH tensor layout: [N, num_sh, 3]
    sh = torch.zeros((N, num_sh, 3), device=device)
    sh[:, 0, :] = torch.from_numpy(f_dc).to(device)

    return means, quats, scales, opacities, sh




def render_gs_pic(means, quats, scales, opacities, sh, T, K, W, H):

    T = get_gs_viewmat(T)

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
        render_mode="RGB+ED",
    )

    img   = image[0, ..., :-1].detach().cpu().numpy() 
    depth = image[0, ..., -1].detach().cpu().numpy() 

    return img, depth






def get_gs_viewmat(T, device="cuda"):
    # Taking normal pose (cam relative to wrld) and turn it to (wrld telative to cam)
    T = torch.tensor(T, dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(T)
    return viewmat.unsqueeze(0)







def turn_points_to_o3d(points) :
    
    n = len(points)
    colors = plt.cm.hsv(np.linspace(0, 1, n))[:, :3]
    spheres=[]
    i=-1

    for point in points:
        
        i+=1
        
        # Create sphere mesh
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        
        # Translate sphere to the point location
        sphere.translate(point)
        
        # Assign color to the sphere
        sphere.paint_uniform_color(colors[i])
        
        # Add to list
        spheres.append(sphere)

    # If you want to combine all spheres into one mesh (optional)
    combined_spheres = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        combined_spheres += sphere

    return combined_spheres




# Getting random well distributed points from moge pointcloud
def get_random_points_frm_moge(points, num_samples=25):

    # Flatten to (H*W, 3)
    pts_flat = points.reshape(-1, 3)
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(pts_flat).all(axis=1)
    pts_flat = pts_flat[valid_mask]
    
    if len(pts_flat) < num_samples:
        return pts_flat
    
    # Initialize: start with a random point
    sampled_indices = [np.random.randint(0, len(pts_flat))]
    sampled_points = [pts_flat[sampled_indices[0]]]
    
    # Distance from each point to the nearest sampled point
    min_distances = np.full(len(pts_flat), np.inf)
    
    # Farthest Point Sampling
    for _ in range(num_samples - 1):
        
        # Update distances to nearest sampled point
        last_point = pts_flat[sampled_indices[-1]]
        distances = np.linalg.norm(pts_flat - last_point, axis=1)
        min_distances = np.minimum(min_distances, distances)
        
        # Select the farthest point
        farthest_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_idx)
        sampled_points.append(pts_flat[farthest_idx])
    
    return np.array(sampled_points)




def draw_matches(matches_1, matches_2, init_img, desired_img):
    
    # Copy images
    im1_vis = init_img.copy()
    im2_vis = desired_img.copy()

    # Ensure uint8
    if im1_vis.dtype != np.uint8:
        im1_vis = (im1_vis * 255).astype(np.uint8)
    if im2_vis.dtype != np.uint8:
        im2_vis = (im2_vis * 255).astype(np.uint8)

    # Stack images side by side
    h1, w1 = im1_vis.shape[:2]
    h2, w2 = im2_vis.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = im1_vis
    canvas[:h2, w1:] = im2_vis

    # Draw matches
    for pt1, pt2 in zip(matches_1, matches_2):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1, int(pt2[1])  # shift x for second image
        color = (0, 255, 0)  # green
        cv2.circle(canvas, (x1, y1), 5, color, -1)
        cv2.circle(canvas, (x2, y2), 5, color, -1)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

    return canvas







def get_3d_from_uv(feats, Zs) :
    # I have a list of fezats [u,v]s and a list of corresp Zs
    # Turning uvs to XYZs and x*Z gives X and y*Z gives Y
    Ss = get_Ss_from_uv(feats)
    Ss = np.array(Ss)      # shape (N,2)
    Zs = np.array(Zs)      # shape (N,)
    XYZs = np.column_stack((Ss * Zs[:, None], Zs))
    return XYZs







def remove_edge_features(coords, mask, nbr_features, radius=2):
    H, W = mask.shape
    valid_indices = []

    for i, (u, v) in enumerate(coords):
        if 0 <= u < W and 0 <= v < H:

            u_min = max(u - radius, 0)
            u_max = min(u + radius + 1, W)
            v_min = max(v - radius, 0)
            v_max = min(v + radius + 1, H)

            neighborhood = mask[v_min:v_max, u_min:u_max]

            if not np.any(neighborhood):  # all False
                valid_indices.append(i)

                if len(valid_indices) >= nbr_features:
                    break

    if len(valid_indices) < nbr_features:
        raise ValueError(
            f"Only {len(valid_indices)} valid features found, "
            f"but {nbr_features} were requested."
        )

    return valid_indices
















def main() :

    # Fixed Vars :
    lambda_gain = 0.2
    dt = 0.1
    all_nbr_ftrs = 100
    nbr_features = 10
    # Scale the mesh to meters, for office_2 nearly 1 meter corresp to 4 units
    scale = (1/4) 


    # Load a 3d mesh nearly in meters
    mesh = load_mesh(mesh_path, scale, False)

    # Visualize and choose a real initial pose || use the saved one
    """init_pose = get_cam_pose_from_mesh_view(mesh)
    np.save("init_pose.npy", init_pose)"""
    init_pose = np.load("init_pose.npy")

    # Move init_pose to origin
    mesh.transform(np.linalg.inv(init_pose))
    init_pose = np.eye(4)


    # Take initial mesh pic || load the saved one
    """init_mesh_img, init_depth = render_mesh_pic(mesh, init_pose)
    save_img(init_mesh_img, 0, o3d_frames_path)"""
    init_mesh_img = load_np_img(f"{o3d_frames_path}/0.png")
    plot_img(init_mesh_img, "init_img")


    # Apply moge on the init real img || load ready mogepoints
    """moge_points_o3d, moge_points, moge_colors = get_moge_points(init_mesh_img) # moge scene dist from cam is not accurate
    np.save("moge_points.npy", moge_points)
    np.save("moge_colors.npy", moge_colors)"""
    moge_points = np.load("moge_points.npy")
    moge_colors = np.load("moge_colors.npy")
    moge_points_o3d = o3d.io.read_point_cloud(moge_points_save_path)
    
    # Choosing a des_pose from moge points 
    des_gs_pose = init_pose @ get_homog([1, 1, 0, 0, 0, 0])
    #des_gs_pose = get_cam_pose_from_mesh_view(mesh)


    # Visualize all
    visualize_scene([moge_points_o3d, mesh])  

 
    # Init gaussians from moge_points & Rendering gs initial pose img 
    gaussians_list = init_gaussians_from_points(moge_points, moge_colors, gs_save_path)    
    init_gs_img, init_gs_depth = render_gs_pic(*gaussians_list, T=init_pose, K=intrins_gs, W=CAM_W, H=CAM_H)
    plot_2_imgs(init_gs_img, init_gs_depth, "init_gs_img", "init_gs_depth")
    

    # Render des_gs_img and des_mesh_img
    des_gs_img, _ = render_gs_pic(*gaussians_list, T=des_gs_pose, K=intrins_gs, W=CAM_W, H=CAM_H)
    des_mesh_img, _ = render_mesh_pic(mesh, des_gs_pose)
    des_gs_img = des_mesh_img
    """plot_2_imgs(init_mesh_img, init_gs_img, "init_mesh_img",  "init_gs_img")
    plot_2_imgs(des_mesh_img, des_gs_img, "des_mesh_img",  "des_gs_img")
    plot_2_imgs(init_gs_img, des_gs_img, "init_gs_img",  "des_gs_img")
    save_img(des_mesh_img, "desired", o3d_frames_path)
    save_img(des_gs_img, "desired", gs_frames_path)"""


    # Matching features with Xfeat between the init and des
    xfeat = XFeat()
    all_matches_init, all_matches_des = xfeat.match_xfeat(init_gs_img, des_gs_img)
    
    # Pick first "all_nbr_ftrs" elemnts & turn them to int
    matches_init = (all_matches_init[:all_nbr_ftrs]).astype(int)
    matches_des = (all_matches_des[:all_nbr_ftrs]).astype(int)

    # Drawing all matches
    init_all_mtch_gs_img = draw_matches(matches_init, matches_des, init_gs_img, des_gs_img)

    # Get depth_edge mask and plot
    edge_mask = utils3d.np.depth_map_edge(init_gs_depth, rtol=0.008) # shape H*W
    plot_2_imgs(init_gs_depth, edge_mask, "gs_depth", "depth_edges")

    # Keep "nbr_ftrs" that doesn't belong to the edges
    valid_indices = remove_edge_features(matches_init, edge_mask, nbr_features)
    matches_init = matches_init[valid_indices]
    matches_des = matches_des[valid_indices]
    init_mtch_gs_img = draw_matches(matches_init, matches_des, init_gs_img, des_gs_img)
    plot_2_imgs(init_mtch_gs_img, init_all_mtch_gs_img, "valid_features", "all_features")

    save_img(init_mtch_gs_img, "mtch_gs_img_1", o3d_frames_path)
    plot_img(init_mtch_gs_img, "mtch_gs_img_1")

    # Get s* from matches
    Ss_star = get_Ss_from_uv(matches_des)


    # Starting IBVS loop
    cur_gs_pose = init_pose
    matp_vis = LiveOptimizationVisualizer(init_gs_img)
    try:    
        for i in range(999) :

            # 1 - Render new cur_gs_pic and get its depthmap
            cur_gs_img, cur_gs_depth_map = render_gs_pic(*gaussians_list, T=cur_gs_pose, K=intrins_gs, W=CAM_W, H=CAM_H)
            cur_mesh_img, _ = render_mesh_pic(mesh, cur_gs_pose)
        
            # Apply X-feat
            all_matches_cur, all_matches_des = xfeat.match_xfeat(cur_gs_img, des_gs_img)
            # Pick first "all_nbr_ftrs" elemnts & turn them to int
            matches_cur = (all_matches_cur[:all_nbr_ftrs]).astype(int)
            matches_des = (all_matches_des[:all_nbr_ftrs]).astype(int)
            # Get depth_edge mask and plot
            edge_mask = utils3d.np.depth_map_edge(cur_gs_depth_map, rtol=0.008) # shape H*W
            # Keep "nbr_ftrs" that doesn't belong to the edges
            valid_indices = remove_edge_features(matches_cur, edge_mask, nbr_features)
            matches_cur = matches_cur[valid_indices]
            matches_des = matches_des[valid_indices]
            cur_mtch_gs_img = draw_matches(matches_cur, matches_des, cur_gs_img, des_gs_img)
            save_img(cur_mtch_gs_img, f"cur_mtch_gs_img_{i}", gs_frames_path)

            # 2 - Get cur and des features
            Ss_star = get_Ss_from_uv(matches_des)
            Ss_cur = get_Ss_from_uv(matches_cur)
            Ss_Z_cur = get_feats_depth(matches_cur, cur_gs_depth_map)


            # 3 - Get the error
            errors = getting_errors(Ss_cur, Ss_star) 
            errors = np.asarray(errors, dtype=float).reshape(-1)

            # Print the norm of the error
            norm_of_error = np.linalg.norm(errors) 
            print(f"error {i} :", norm_of_error)


            # 4 - Get the intr matrix nd its pseudo_inv 
            L = get_interaction_matrix(nbr_features, Ss_cur, Ss_Z_cur, 1)
            LinAlgeb.print_mat_condition_number(L)
            L_psinv = get_inter_mat_pseudo_inverse(L)


            # 5 - Choose lambda and calculate V with control law
            if norm_of_error < 0.25 :
                lambda_gain = 0.2
            if norm_of_error < 0.0001 :
                break        
            V = - lambda_gain * (L_psinv @ errors)  
           
            # 6 - Update cur_cam_pose and update visualization
            cur_gs_pose = update_cam_pose(cur_gs_pose, V, dt)
            matp_vis.update(i, cur_mtch_gs_img, cur_gs_img, cur_mesh_img, des_gs_img, V, norm_of_error)
   
    
    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            matp_vis.close()
            os._exit(0)

    


if __name__ == "__main__":
    main()