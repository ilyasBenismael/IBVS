import pycolmap
import torch
import numpy as np
from plyfile import PlyData
from utils.lin_algeb import LinAlgeb
import matplotlib.pyplot as plt
from gsplat import rasterization
import cv2
import math
from datetime import datetime
from utils.visualizer import LiveOptimizationVisualizer
import open3d as o3d







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

    print("=== Means (first point) ===")
    print(means[0])

    # -----------------------
    # Scales (log → real)
    # -----------------------
    scales_log = torch.stack([
        torch.from_numpy(ply["scale_0"]),
        torch.from_numpy(ply["scale_1"]),
        torch.from_numpy(ply["scale_2"]),
    ], dim=1).float().to(device)
    print("=== Scales log (first point) ===")
    print(scales_log[0])

    scales = torch.exp(scales_log)
    print("=== Scales exp (first point) ===")
    print(scales[0])

    # -----------------------
    # Rotation (normalize quaternion)
    # -----------------------
    quats = torch.stack([
        torch.from_numpy(ply["rot_0"]),
        torch.from_numpy(ply["rot_1"]),
        torch.from_numpy(ply["rot_2"]),
        torch.from_numpy(ply["rot_3"]),
    ], dim=1).float().to(device)

    print("=== Quats raw (first point) ===")
    print(quats[0])

    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    print("=== Quats normalized (first point) ===")
    print(quats[0])

    # -----------------------
    # Opacity (inverse sigmoid → alpha)
    # -----------------------
    opacity_param = torch.from_numpy(ply["opacity"]).float().to(device)
    print("=== Opacity param (logit) (first point) ===")
    print(opacity_param[0])

    opacities = torch.sigmoid(opacity_param)
    print("=== Opacity alpha (sigmoid) (first point) ===")
    print(opacities[0])

    # -----------------------
    # Spherical Harmonics
    # -----------------------
    f_dc = torch.stack([
        torch.from_numpy(ply["f_dc_0"]),
        torch.from_numpy(ply["f_dc_1"]),
        torch.from_numpy(ply["f_dc_2"]),
    ], dim=1)
    print("=== SH f_dc (first point) ===")
    print(f_dc[0])
    
    # Compute number of rest coefficients
    ply_data = ply.data  # <-- access the structured array
    rest_keys = [k for k in ply_data.dtype.names if k.startswith("f_rest_")]

    f_rest = torch.stack([
        torch.from_numpy(ply_data[k]) for k in rest_keys
    ], dim=1)

    print("=== SH f_rest (first 5 coefficients, first point) ===")
    print(f_rest[0, :15])  # first 5 rest coeffs (5*3=15)

    sh = torch.cat([f_dc, f_rest], dim=1)   # (N, num_coeffs*3)
    sh = sh.view(N, -1, 3).float().to(device)

    print("=== SH final shape ===")
    print(sh.shape)
    print("=== SH first point all coefficients ===")
    print(sh[0])


    return means, quats, scales, opacities, sh









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

    # 1 - loading my scene :
    means, quats, scales, opacities, sh = load_gaussians_from_ply(path)

    return

    # 2 - defining desired cam T 
    T_1 = get_homog([-1, 2, 5, 0, 0, 0])
    T_2 = get_homog([0, 0, 0, 0, -1.1*np.pi, 0])
    T_3 = get_homog([0, 0, 0, 0, 0, -np.pi])
    T_4 = get_homog([0, 0, 0, -np.pi/5, 0, 0])
    des_T = T_1 @ T_2 @ T_3 @ T_4 #homog of camera frame relatively to world frame
    des_viewmat = make_viewmat(des_T) #tensor of homog of world frame relatively to camera frame

    # 3 defining initial cam T
    T_1 = get_homog([0.7, -0.5, -0.6, np.pi/25, np.pi/20, 0])
    cur_T = des_T @ T_1

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
        for i in range(999):

            # 1 - taking pic from cur_T
            cur_viewmat = make_viewmat(cur_T) 
            cur_img = render_view(means, quats, scales, opacities, sh, cur_viewmat, Ks, CAM_W, CAM_H)
            gray_cur = 0.299 * cur_img[:, :, 0] + 0.587 * cur_img[:, :, 1] + 0.114 * cur_img[:, :, 2]
            S = gray_cur.flatten()
            if(i==0):
                #making our matp visualizer
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
            if cost > 20000 :  
                max_val = 0.1 
                mu = 999999999 
            elif cost > 5000 :  
                max_val = 0.1 
                mu=0.0000001   
            elif cost > 500 :  
                max_val = 0.05  
                mu=0.0000001 
            elif cost > 50 :  
                max_val = 0.01  
                mu=0.0000001 
            elif cost > 15 : 
                max_val = 0.01  
            else : 
                break 
                    
                                
            # 5 - Compute V with GN or LM 
            V = -lamda * np.linalg.solve(Ls.T @ Ls + mu * np.diag(np.diag(Ls.T @ Ls)), Ls.T @ diff)      
            #V = -lamda * np.linalg.pinv(Ls) @ diff
            V = scale_by_max(V, max_val)

            
            # 6 - Update camera pose & update matplotlib vis data
            cur_T = update_cam_pose(cur_T, V, dt)

            save_img(cur_img, i, "frames")
            matp_vis.update(i, cur_img, cur_img, current_diff_img, V, cost)

        matp_vis.close()


    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting loop cleanly.")
        matp_vis.close()





if __name__ == "__main__":
    main()