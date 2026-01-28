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





def load_gaussians_from_ply(ply_path, device="cuda"):
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    N = v.shape[0]
    print(N)


    # Positions
    means = torch.stack([
        torch.from_numpy(v["x"]),
        torch.from_numpy(v["y"]),
        torch.from_numpy(v["z"]),
    ], dim=1).float()


    # Rotations
    quats = torch.stack([
        torch.from_numpy(v["rot_0"]),
        torch.from_numpy(v["rot_1"]),
        torch.from_numpy(v["rot_2"]),
        torch.from_numpy(v["rot_3"]),
    ], dim=1).float()

    quats = quats / quats.norm(dim=1, keepdim=True)


    # Scales
    scales_log = torch.stack([
        torch.from_numpy(v["scale_0"]),
        torch.from_numpy(v["scale_1"]),
        torch.from_numpy(v["scale_2"]),
    ], dim=1).float()

    scales = torch.exp(scales_log)
    scales = scales.clamp(1e-4, 0.05)


    # Opacity
    opacities = torch.sigmoid(
        torch.from_numpy(v["opacity"]).float()
    )


    # SH coefficients
    f_rest_cols = [k for k in v.dtype.names if k.startswith("f_rest_")]
    num_rest = len(f_rest_cols) // 3
    num_coeffs = 1 + num_rest

    sh = torch.zeros((N, num_coeffs, 3), dtype=torch.float32)

    sh[:, 0, 0] = torch.from_numpy(v["f_dc_0"])
    sh[:, 0, 1] = torch.from_numpy(v["f_dc_1"])
    sh[:, 0, 2] = torch.from_numpy(v["f_dc_2"])

    for i in range(num_rest):
        sh[:, i + 1, 0] = torch.from_numpy(v[f"f_rest_{3*i+0}"])
        sh[:, i + 1, 1] = torch.from_numpy(v[f"f_rest_{3*i+1}"])
        sh[:, i + 1, 2] = torch.from_numpy(v[f"f_rest_{3*i+2}"])

    return (
        means.to(device),
        quats.to(device),
        scales.to(device),
        opacities.to(device),
        sh.to(device),
    )





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








def get_gaussians_from_points(
    xyz,
    rgb,
    ply_path,
    scale=0.01,
    opacity=1,
    device="cuda",
    sh_degree=2,
):

    xyz_flat = xyz.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    N = xyz_flat.shape[0]

    # Getting gaussian properties for gsplat rasterization

    means = torch.from_numpy(xyz_flat).float().to(device)

    quats = torch.zeros((N, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # identity quaternion (w,x,y,z)

    scales = torch.full((N, 3), scale, dtype=torch.float32, device=device)

    opacities = torch.full((N,), opacity, dtype=torch.float32, device=device)

    num_coeffs = (sh_degree + 1) ** 2
    sh = torch.zeros((N, num_coeffs, 3), dtype=torch.float32, device=device)
    sh[:, 0, :] = torch.from_numpy(rgb_flat).float().to(device)

    

    # Storing PLy______________________

    # scale must be stored in log-space
    scale = np.log(scale)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),

        ("rot_0", "f4"), ("rot_1", "f4"),
        ("rot_2", "f4"), ("rot_3", "f4"),

        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),

        ("opacity", "f4"),

        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]

    for i in range(1, num_coeffs):
        for c in range(3):
            dtype.append((f"f_rest_{3*(i-1)+c}", "f4"))

    data = np.empty(N, dtype=dtype)

    # positions
    data["x"] = xyz_flat[:, 0]
    data["y"] = xyz_flat[:, 1]
    data["z"] = xyz_flat[:, 2]

    # rotation
    data["rot_0"] = 1.0
    data["rot_1"] = 0.0
    data["rot_2"] = 0.0
    data["rot_3"] = 0.0

    # scale (log)
    data["scale_0"] = scale
    data["scale_1"] = scale
    data["scale_2"] = scale

    # opacity 
    data["opacity"] = opacity

    # SH colors
    data["f_dc_0"] = rgb_flat[:, 0]
    data["f_dc_1"] = rgb_flat[:, 1]
    data["f_dc_2"] = rgb_flat[:, 2]

    # higher SH = 0
    for i in range(1, num_coeffs):
        for c in range(3):
            data[f"f_rest_{3*(i-1)+c}"] = 0.0

    PlyData([PlyElement.describe(data, "vertex")]).write(ply_path)

    print(
        f"Saved {ply_path} | "
        f"{N} Gaussians | "
        f"scale={scale} | "
        f"opacity={opacity:.3f} | "
        f"SH degree={sh_degree}"
    )

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


    means, quats, scales, opacities, sh = get_gaussians_from_points(
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