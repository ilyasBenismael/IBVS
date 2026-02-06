import torch
import numpy as np
from plyfile import PlyData, PlyElement
from utils.lin_algeb import LinAlgeb
import matplotlib.pyplot as plt
from gsplat import rasterization
import cv2
import math
from datetime import datetime
from utils.visualizer import LiveOptimizationVisualizer
from moge.model.v2 import MoGeModel
import open3d as o3d
from PIL import Image











#_____ parameters

CAM_W, CAM_H = 700, 600
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

dt = 0.4
lamda = 0.1
path = "scenes/desk.ply"
img_path = "imgs/room.jpeg"



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
    clrs  = colors.reshape(-1, 3)
    valid   = mask.reshape(-1) > 0

    # Create Open3D point cloud with only valid points
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(pts[valid])
    points_o3d.colors = o3d.utility.Vector3dVector(clrs[valid])
    return points_o3d, points_world, points_cam, colors





def make_viewmat(T_wc, device="cuda"):
    T_wc = torch.tensor(T_wc, dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(T_wc)
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




def compute_image_interaction_matrix(grad_Ix, grad_Iy):    

    # Create 2d of just u coords nd 2d of just v coords, both (H, W)
    u_coords, v_coords = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))  
    
    # get a 2d of x coords nd y coords in meters , both (H, W)
    x = (u_coords - CX) / FX
    y = (v_coords - CY) / FY
    
    # get a copy of depth with 0s as 100
    Z = np.full((CAM_H, CAM_W, 1), 4)
    
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









def make_ply_from_points(xyz, rgb, ply_path, scale=0.01, alpha=0.95, sh_degree=2, device="cuda"):
    
    # Flatten points nd colors
    xyz_flat = xyz.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    N = xyz_flat.shape[0]

    # make sure colors are normalized to 0-1
    if rgb_flat.max() > 1.0:
        rgb_flat = rgb_flat / 255.0
    rgb_flat = np.clip(rgb_flat, 0.0, 1.0)

    # Opacity logit
    opacity_logit = alpha
                                                                                                    
    """
    eps = 1e-6
    alpha_clamped = np.clip(alpha, eps, 1.0 - eps)
    opacity_logit = np.log(alpha_clamped / (1 - alpha_clamped))
    """
    
    # Scale in log-space
    scale_log = np.log(scale)

    # SH coefficients
    num_coeffs = (sh_degree + 1) ** 2
    f_dc = rgb_flat.copy()                      # 0th-order SH
    f_rest = np.zeros((N, num_coeffs - 1, 3))  # remaining SH coefficients all zeros
    sh_all = np.concatenate([f_dc[:, np.newaxis, :], f_rest], axis=1)  # shape (N, num_coeffs, 3)

    # Define PLY dtype
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    for i in range(1, num_coeffs):
        for c in range(3):
            dtype.append((f"f_rest_{3*(i-1)+c}", "f4"))

    # Fill PLY data
    data = np.empty(N, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz_flat.T
    data["nx"] = data["ny"] = data["nz"] = 0.0
    data["rot_0"] = 1.0
    data["rot_1"] = data["rot_2"] = data["rot_3"] = 0.0
    data["scale_0"] = data["scale_1"] = data["scale_2"] = scale_log
    data["opacity"] = -1.8042
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]

    for i in range(1, num_coeffs):
        for c in range(3):
            data[f"f_rest_{3*(i-1)+c}"] = f_rest[:, i-1, c]

    # Write PLY
    PlyData([PlyElement.describe(data, "vertex")]).write(ply_path)
    print(f"Saved reference PLY: {ply_path} | {N} Gaussians | scale={scale} | alpha={alpha} | SH degree={sh_degree}")






    # -----------------------------
    # Debug prints for first few points
    # -----------------------------
    num_print = min(5, N)
    for idx in range(num_print):
        print(f"\n=== Gaussian {idx} ===")
        print("XYZ:", xyz_flat[idx])
        print("RGB normalized:", rgb_flat[idx])
        print("Opacity logit:", opacity_logit)
        print("Scale log:", scale_log)
        print("Rotation quaternion:", [1.0, 0.0, 0.0, 0.0])
        print("SH f_dc:", f_dc[idx])
        print("SH f_rest (all):", f_rest[idx])
        print("All SH concatenated:", sh_all[idx])

    # Return runtime tensors for further processing if needed
    means = torch.from_numpy(xyz_flat).float().to(device)
    scales = torch.full((N, 3), scale, dtype=torch.float32, device=device)
    quats = torch.zeros((N, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((N,), alpha, dtype=torch.float32, device=device)
    sh = torch.from_numpy(sh_all).float().to(device)

    return means, quats, scales, opacities, sh







def init_gaussians_from_points(
    xyz,
    rgb,
    ply_path,
    scale=0.01,
    alpha=0.1,
    sh_degree=2,
    device="cuda",
):
    """
    Initialize 3D Gaussians from colored point clouds.

    This function:
    1) Converts RGB → SH (0th order) using GS convention
    2) Writes a GS-compatible PLY file
    3) Creates runtime tensors compatible with gsplat / Inria code

    IMPORTANT:
    - Behavior must match Inria initialization exactly
    - No rendering logic is here, only parameter initialization
    """

    # ==========================================================
    # Spherical Harmonics constant (from Gaussian Splatting paper)
    # ==========================================================
    # SH basis Y_0^0 = C0
    # Renderer reconstructs:
    #   RGB = C0 * f_dc + 0.5
    C0 = 0.28209479177387814

    # ==========================================================
    # Input preparation
    # ==========================================================
    # Flatten input arrays
    xyz = xyz.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    N = xyz.shape[0]

    # Ensure RGB is normalized to [0, 1]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    # ==========================================================
    # RGB → SH (0th order coefficient only)
    # ==========================================================
    # In Gaussian Splatting, colors are stored in SH space.
    # The DC (0th order) coefficient encodes view-independent color.
    #
    # Inria / gsplat convention:
    #   f_dc = (RGB - 0.5) / C0
    #
    # Renderer later reconstructs:
    #   RGB = C0 * f_dc + 0.5
    #
    # This ensures zero-centered SH coefficients and stable optimization.
    f_dc = (rgb - 0.5) / C0

    # Number of SH coefficients:
    # degree=2 → (2+1)^2 = 9
    num_sh = (sh_degree + 1) ** 2

    # ==========================================================
    # Opacity initialization
    # ==========================================================
    # Gaussian Splatting optimizes opacity in logit space.
    #
    # Forward rendering uses:
    #   alpha = sigmoid(opacity_logit)
    #
    # So we store the inverse-sigmoid here.
    eps = 1e-6
    alpha = np.clip(alpha, eps, 1 - eps)
    opacity_logit = np.log(alpha / (1 - alpha))

    # ==========================================================
    # Scale initialization
    # ==========================================================
    # GS stores Gaussian scale in log-space:
    #   scale = exp(scale_log)
    #
    # We initialize isotropic Gaussians (same scale in x,y,z)
    scale_log = np.log(scale)

    # ==========================================================
    # PLY file structure (GS-compatible)
    # ==========================================================
    # This layout exactly matches what gsplat / Inria loaders expect.
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),          # Gaussian center
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),       # Unused normals
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),  # SH DC
        ("opacity", "f4"),                              # Opacity logit
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),  # log-scales
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),  # quaternion
    ]

    # Higher-order SH coefficients (initialized to zero)
    for i in range(num_sh - 1):
        for c in range(3):
            dtype.append((f"f_rest_{3*i + c}", "f4"))

    data = np.empty(N, dtype=dtype)

    # ==========================================================
    # Fill PLY data
    # ==========================================================
    # Positions
    data["x"], data["y"], data["z"] = xyz.T

    # Normals are unused by GS → set to zero
    data["nx"] = data["ny"] = data["nz"] = 0.0

    # SH DC coefficients
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]

    # Opacity logit
    data["opacity"] = opacity_logit

    # Isotropic scale (log-space)
    data["scale_0"] = scale_log
    data["scale_1"] = scale_log
    data["scale_2"] = scale_log

    # Identity rotation quaternion
    # (w, x, y, z) = (1, 0, 0, 0)
    data["rot_0"] = 1.0
    data["rot_1"] = 0.0
    data["rot_2"] = 0.0
    data["rot_3"] = 0.0

    # Higher-order SH coefficients start at zero
    for i in range(num_sh - 1):
        for c in range(3):
            data[f"f_rest_{3*i + c}"] = 0.0

    # Write PLY to disk
    PlyData([PlyElement.describe(data, "vertex")]).write(ply_path)

    # ==========================================================
    # Runtime tensors for gsplat
    # ==========================================================
    # These tensors are used directly by the renderer / optimizer.
    means = torch.from_numpy(xyz).float().to(device)
    scales = torch.full((N, 3), scale, device=device)
    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0  # identity rotation
    opacities = torch.full((N,), alpha, device=device)

    # SH tensor layout: [N, num_sh, 3]
    sh = torch.zeros((N, num_sh, 3), device=device)
    sh[:, 0, :] = torch.from_numpy(f_dc).to(device)

    # ==========================================================
    # Debug print (single Gaussian)
    # ==========================================================
    print("=============== GS INIT DEBUG ===============")
    print("XYZ:", xyz[0])
    print("RGB:", rgb[0])
    print("f_dc (SH space):", f_dc[0])
    print("Reconstructed RGB:", C0 * f_dc[0] + 0.5)
    print("Opacity:", alpha, "| logit:", opacity_logit)
    print("Scale:", scale, "| log:", scale_log)
    print("============================================")
    print(f"Saved PLY: {ply_path} | {N} Gaussians | SH degree={sh_degree}")

    return means, quats, scales, opacities, sh













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






#__________________________________________________________________________________


def main() :
    
    # Defining desired cam T 
    des_T = get_homog([0, 0, 0, 0, 0, 0])
    des_viewmat = make_viewmat(des_T) #tensor of homog of world frame relatively to camera frame

    # defining initial cam T
    T_1 = get_homog([0.3, -0.2, 0, 0, 0, 0]) #near
    #T_1 = get_homog([2.2, 0, -2, 0, 0, 0]) #far
    cur_T = des_T @ T_1

    #getting 3dpoints in world frame and translating them to cur cam nd getting back colors nd points in world
    points_o3d, points_world, points_cam, colors = load_points(img_path, cur_T)


    means, quats, scales, opacities, sh = init_gaussians_from_points(
        xyz=points_world,
        rgb=colors,
        ply_path="scenes/init_gaussians.ply",
    )



    # 4 - rendering desired image
    t1 = datetime.now()    
    des_img = render_view(means, quats, scales, opacities, sh, des_viewmat, Ks, CAM_W, CAM_H)
    t2 = datetime.now()
    diff_ms = (t2 - t1).total_seconds() * 1000
    print("rendering time in ms : ", diff_ms)

    save_img(des_img, "desired img", "frames")
    plot_img(des_img, "des_img")
    gray_des = 0.299 * des_img[:, :, 0] + 0.587 * des_img[:, :, 1] + 0.114 * des_img[:, :, 2]
    S_star = gray_des.flatten()



    try:
        for i in range(500):

            # 1 - taking pic from cur_T
            cur_viewmat = make_viewmat(cur_T) 
            cur_img = render_view(means, quats, scales, opacities, sh, cur_viewmat, Ks, CAM_W, CAM_H)
            gray_cur = 0.299 * cur_img[:, :, 0] + 0.587 * cur_img[:, :, 1] + 0.114 * cur_img[:, :, 2]
            S = gray_cur.flatten()


            if(i==0):
                plot_img(cur_img, "initial img")         
                matp_vis = LiveOptimizationVisualizer(des_img, cur_img)

            # 2 - Compute the cost with des_img
            diff = S - S_star
            cost = diff.T @ diff
            print(f"Cost {i} :", cost)
            current_diff_img = compute_grayscale_difference(gray_cur, gray_des)

            # 3 - Compute Gradient and Ls 
            grad_Ix, grad_Iy = get_grads_visp(gray_cur)
            Ls = compute_image_interaction_matrix(grad_Ix, grad_Iy)

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
            cur_T = update_cam_pose(cur_T, V, dt)

            save_img(cur_img, i, "gs_frames")
            matp_vis.update(i, cur_img, cur_img, current_diff_img, V, cost)

        matp_vis.close()


    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting loop cleanly.")
        matp_vis.close()





if __name__ == "__main__":
    main()

 