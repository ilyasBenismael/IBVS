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




def get_moge_points(img, threshold=0.01):

    device = "cuda"
    model = MoGeModel.from_pretrained(
        "Ruicheng/moge-2-vitl-normal"
    ).to(device)
    model.eval()

  
    if img.max() > 1.0:
        img = img / 255.0

    # img is already in [0,1], RGB
    image = torch.from_numpy(img) \
        .float() \
        .to(device) \
        .permute(2, 0, 1)

    output = model.infer(image)

    points = output["points"].cpu().numpy()
    depth  = output["depth"].cpu().numpy()
    mask   = output["mask"].cpu().numpy()

    edge_mask = utils3d.np.depth_map_edge(depth, rtol=threshold)
    mask_cleaned = mask & (~edge_mask)

    # ---- IMPORTANT PART ----
    colors = img.astype(np.float64)
    pts_flat    = points.reshape(-1, 3).astype(np.float64)
    colors_flat = colors.reshape(-1, 3)
    valid_flat  = mask_cleaned.reshape(-1)

    final_points = pts_flat[valid_flat]
    final_colors = colors_flat[valid_flat]

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(final_points)
    o3d_points.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(moge_points_save_path, o3d_points)

    return o3d_points, final_points, final_colors






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





def get_camera_pose_from_mesh(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Set Camera View", width=800, height=800)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    
    print("Navigate to desired view.")
    print("Close window when done.")
    
    vis.run()
    
    # Get camera parameters
    vc = vis.get_view_control()
    cam_params = vc.convert_to_pinhole_camera_parameters()
    
    # Extract pose: camera to world
    T_wc = cam_params.extrinsic  # world to camera
    T_cw = np.linalg.inv(T_wc)    # camera to world
    
    vis.destroy_window()
    return T_cw




    
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
    cam_params.intrinsic = intrins2_o3d
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







def start_dvs_loop(iterations, des_img) : 

    for i in range(iterations) :
        
        if i== 0:
            gray_des = 0.299 * des_img[:, :, 0] + 0.587 * des_img[:, :, 1] + 0.114 * des_img[:, :, 2]
            S_star = gray_des.flatten()







def get_gs_viewmat(T, device="cuda"):
    # Taking normal pose (cam relative to wrld) and turn it to (wrld telative to cam)
    T = torch.tensor(T, dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(T)
    return viewmat.unsqueeze(0)






def init_gaussians_from_points(
    xyz,
    rgb,
    ply_path,
    scale=0.007,
    alpha=0.1,
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
    opacity_logit = np.log(alpha / (1 - alpha))

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









#____________________________________________________________________________________




def main() :
    
    # Scale the mesh to meters, for office_2 nearly 1 meter corresp to 4 units
    scale = (1/4) 

    # Load a 3d mesh nearly in meters
    mesh = load_mesh(mesh_path, scale, False)

    # Visualize and choose an initial pose || use the saved one
    """init_pose = get_cam_pose_from_mesh_view(mesh)
    np.save("init_pose.npy", init_pose)"""
    init_pose = np.load("init_pose.npy")

    # Move init_pose to origin
    mesh.transform(np.linalg.inv(init_pose))
    init_pose = np.eye(4)


    # Take initial mesh pic || load the saved one
    """mesh_init_img, init_depth = render_mesh_pic(mesh, init_pose)
    save_img(mesh_init_img, 0, o3d_frames_path)"""
    mesh_init_img = load_np_img(f"{o3d_frames_path}/0.png")
    plot_img(mesh_init_img, "init_img")


    # Apply moge on the init real img || load ready mogepoints
    """points_o3d, moge_points, moge_colors = get_moge_points(mesh_init_img) # moge scene dist from cam is not accurate
    np.save("moge_points.npy", moge_points)
    np.save("moge_colors.npy", moge_colors)"""
    moge_points = np.load("moge_points.npy")
    moge_colors = np.load("moge_colors.npy")
    points_o3d = o3d.io.read_point_cloud(moge_points_save_path)
    visualize_scene([points_o3d, mesh])    

    # Getting a des_pose from real mesh and plotting it
    gs_des_pose = get_cam_pose_from_mesh_view(points_o3d)

    # Init gaussians from moge_points & Rendering gs initial pose img 
    gaussians_list = init_gaussians_from_points(moge_points, moge_colors, gs_save_path)
    init_viewmat = get_gs_viewmat(init_pose)
    gs_init_img = render_gs_pic(*gaussians_list, T=init_viewmat, K=intrins_gs, W=CAM_W, H=CAM_H)
    save_img(gs_init_img, 0, gs_save_path)
    
    # Render des_img from gaussians and mesh
    des_viewmat = get_gs_viewmat(gs_des_pose)
    gs_des_img = render_gs_pic(*gaussians_list, T=des_viewmat, K=intrins_gs, W=CAM_W, H=CAM_H)
    mesh_des_img, _  = render_mesh_pic(mesh, gs_des_pose)  
    plot_2_imgs(mesh_des_img, gs_des_img, "mesh_des_img",  "gs_des_img")
    plot_2_imgs(gs_init_img, gs_des_img, "gs_init_img",  "gs_des_img")
    save_img(mesh_des_img, "desired", o3d_frames_path)
    save_img(gs_des_img, "desired", gs_frames_path)

    
    return

    

    start_dvs_loop(iterations = 999, des_img = mesh_des_img,  )
    # using a very clean code
    # apply dvs, dyna and stat ibvs on both scenes (navigate and render in both scenes use correct depth and avg depth)
    # turn to gaussians, and redo same thing
    # turn real poses to colmap and improve gaussians


    #visualize the mesh and return des_pose 
    #render des pose in o3d nd moge to compare 



    return

    

    # intialize trajectory scene : the visualizer with frames, mesh and line
    camera_centers = []
    trajectory_line = o3d.geometry.LineSet()
    o3d_vis = o3d.visualization.Visualizer()
    #initialize_traject_visualizer(o3d_vis, trajectory_line, des_cam_axis, cur_cam_axis, des_extrins, mesh)
    prev_cur_frame = None




    #____________________________________________________________________________________

    # Closed-loop
    try:
        for i in range(999):

            # 0 - Update visualizer of trajects
            #prev_cur_frame, scene_img = update_traject_visualizer(i, o3d_vis, trajectory_line, camera_centers, cur_extrins, prev_cur_frame)

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
            #matp_vis.update(i, scene_img, cur_img, current_diff_img, V,  cost)

        matp_vis.close()
        o3d_vis.close()

    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting loop cleanly.")
        matp_vis.close()
        o3d_vis.close()
        os._exit(0)



if __name__ == "__main__":
    main()