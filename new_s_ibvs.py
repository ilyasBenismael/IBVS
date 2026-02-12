import open3d as o3d
import numpy as np
from utils.lin_algeb import LinAlgeb
import cv2
import matplotlib.pyplot as plt
import math
import os
from plyfile import PlyData, PlyElement
from PIL import Image
from datetime import datetime
from utils.visualizer import LiveOptimizationVisualizer
import torch
from typing import Tuple, Sequence, List, Union
from moge.model.v2 import MoGeModel
from gsplat import rasterization
import utils3d 








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
    device = "cuda"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()

    if img.max() > 1.0:
        img = img / 255.0

    image = torch.from_numpy(img).float().to(device).permute(2, 0, 1)
    output = model.infer(image)

    points = output["points"].cpu().numpy()  # (H, W, 3)
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()

    edge_mask = utils3d.np.depth_map_edge(depth, rtol=threshold)
    mask_cleaned = mask & (~edge_mask)

    colors = img.astype(np.float64)
    
    # For Open3D visualization - save filtered points
    pts_flat = points.reshape(-1, 3).astype(np.float64)
    colors_flat = colors.reshape(-1, 3)
    valid_flat = mask_cleaned.reshape(-1)

    final_points = pts_flat[valid_flat]
    final_colors = colors_flat[valid_flat]

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(final_points)
    o3d_points.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(moge_points_save_path, o3d_points)

    # Return FULL arrays (H, W, 3) for indexing by pixel coordinates
    return o3d_points, points, colors  # ← Changed from final_points, final_colors






    
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
    return uv_list






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
        
        if(depth_map[f[1],f[0]]>100) :
            depth_map[f[1],f[0]] = 100

        if(depth_map[f[1],f[0]] < 0) :
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







def turn_points_to_o3d(points) :
    n = len(points)
    colors = plt.cm.hsv(np.linspace(0, 1, n))[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd




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








def main() :

    # Fixed Vars :
    lambda_gain = 0.3
    dt = 0.1
    nbr_features = 6
    
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
    
    # Visualize all
    visualize_scene([moge_points_o3d, mesh])  

 
    """# Init gaussians from moge_points & Rendering gs initial pose img 
    #gaussians_list = init_gaussians_from_points(moge_points, moge_colors, gs_save_path)
    #init_viewmat = get_gs_viewmat(init_pose)
    #init_moge_img, gs_init_depth = render_gs_pic(*gaussians_list, T=init_viewmat, K=intrins_gs, W=CAM_W, H=CAM_H)"""


    # choosing a des_pose from moge points 
    des_moge_pose = get_cam_pose_from_mesh_view(moge_points_o3d)

    # render initial moge pic
    init_moge_img, _ = render_mesh_pic(moge_points_o3d, init_pose)
    

    # Render des_img from moge and mesh
    #des_viewmat = get_gs_viewmat(des_moge_pose)
    #des_moge_img, _ = render_gs_pic(*gaussians_list, T=des_viewmat, K=intrins_gs, W=CAM_W, H=CAM_H)
    des_mesh_img, _  = render_mesh_pic(mesh, des_moge_pose) 
    des_moge_img, _ = render_mesh_pic(moge_points_o3d, des_moge_pose)
    gray_des_moge_img = 0.299 * des_moge_img[:, :, 0] + 0.587 * des_moge_img[:, :, 1] + 0.114 * des_moge_img[:, :, 2] #just for diff visualization
    plot_2_imgs(init_mesh_img, init_moge_img, "init_mesh_img",  "init_moge_img")
    plot_2_imgs(des_mesh_img, des_moge_img, "des_mesh_img",  "des_moge_img")
    plot_2_imgs(init_moge_img, des_moge_img, "init_moge_img",  "des_moge_img")
    save_img(des_mesh_img, "desired", o3d_frames_path)
    save_img(des_moge_img, "desired", gs_frames_path)
    


    """# Getting top n matched features
    des_kp, des_desc = get_sift_features(des_moge_img)
    cur_kp, cur_des = get_sift_features(init_moge_img)
    cur_feats, des_feats = get_matches(nbr_features, cur_kp, cur_des, des_kp, des_desc, init_moge_img, des_moge_img, show=True)

    # Get 3d points in wf corresponding to the features using (curfeats nd curimgs)   
    fixed_points_wf = []
    for j in range(len(cur_feats)) :
        fixed_points_wf.append(moge_points[cur_feats[j][1], cur_feats[j][0]])
        # Points_world (H, W, 3(xyz)) in world frame corresponding to original init image, and cur_feats is a list of [u,v] (of ftrs) from rendered image
    
    fixed_points_wf = filter_valid_points(fixed_points_wf)"""
    


    # Getting random n points from moge points
    all_candidate_points = get_random_points_frm_moge(moge_points, num_samples=250)

    # Starting IBVS loop
    cur_moge_pose = init_pose
    matp_vis = LiveOptimizationVisualizer(init_moge_img)
    V_prev = np.zeros(6)


    try:    
        for i in range(999) :

            # 1 - Get first n valid elmnts(in front of both cams) frm candidates and optio visualize
            valid_points_wf = filter_valid_points(all_candidate_points, init_pose, des_moge_pose, nbr_features)

            # 2 - get points in cur and des cam and their ss features
            valid_points_cf = LinAlgeb.transform_points_to_cam(valid_points_wf, des_moge_pose)
            Ss_star = get_Ss_from_points(valid_points_cf)
            pxls_star = get_uv_from_Ss(Ss_star)
            valid_points_cf = LinAlgeb.transform_points_to_cam(valid_points_wf, cur_moge_pose)
            Ss = get_Ss_from_points(valid_points_cf)
            pxls = get_uv_from_Ss(Ss)

            # 3 - Rendering current img, nd get depth exprssd in camera frame with cam frame units 
            cur_moge_img, cur_depth_map = render_mesh_pic(moge_points_o3d, cur_moge_pose)
            gray_cur_moge_img = 0.299 * cur_moge_img[:, :, 0] + 0.587 * cur_moge_img[:, :, 1] + 0.114 * cur_moge_img[:, :, 2]
            cur_moge_img_mtchs, des_moge_img_mtchs = draw_matches(cur_moge_img, des_moge_img, pxls, pxls_star)
            save_img(cur_moge_img_mtchs,i, gs_frames_path)

            # Render frm same pose the mesh pic
            cur_mesh_img, _ = render_mesh_pic(mesh, cur_moge_pose)
            
            # 4 - Get corresp uvs and their depth in meters
            cur_feats_uv = get_uv_from_Ss(Ss)
            Ss_Z = get_feats_depth(cur_feats_uv, cur_depth_map)

            # 5 - Getting list of errors nd reshaping it
            errors = getting_errors(Ss, Ss_star) 
            errors = np.asarray(errors, dtype=float).reshape(-1)

            # 6 - Print the norm of the error and get the diff of imgs
            norm_of_error = np.linalg.norm(errors) 
            current_diff_img = compute_grayscale_difference(gray_cur_moge_img, gray_des_moge_img)
            print(f"error {i} :", norm_of_error)

            
            # 7 - Get the intr matrix nd its pseudo_inv 
            L = get_interaction_matrix(nbr_features, Ss, Ss_Z, 1)
            LinAlgeb.print_mat_condition_number(L)
            L_psinv = get_inter_mat_pseudo_inverse(L)


            # 8 - Control law
            if norm_of_error < 0.25 :
                lambda_gain = 0.1
            if norm_of_error < 0.0001 :
                break        
            V = - lambda_gain * (L_psinv @ errors)  
           
            # 9 - Apply the velocity for dt and update cur_cam_pose and update visualization
            cur_moge_pose = update_cam_pose(cur_moge_pose, V, dt)
            matp_vis.update(i, current_diff_img, cur_moge_img_mtchs, cur_mesh_img, des_moge_img_mtchs, V, norm_of_error)

            # 10 - Updating the valid points (probably we will return same old first valid ones )
            valid_points_wf = filter_valid_points(all_candidate_points, cur_moge_pose, des_moge_pose, nbr_features)

    
    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            matp_vis.close()
            os._exit(0)

    


if __name__ == "__main__":
    main()