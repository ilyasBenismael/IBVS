import math
from typing import Tuple, Sequence, List
import numpy as np




class LinAlgeb :

    # --------- Basic axis rotations (rotation around own axis) ----------
    
    @staticmethod
    def Rx(a: float) -> List[List[float]]:
        ca, sa = math.cos(a), math.sin(a)
        return [[1, 0, 0],
                [0, ca, -sa],
                [0, sa,  ca]]
    
    @staticmethod
    def Ry(a: float) -> List[List[float]]:
        ca, sa = math.cos(a), math.sin(a)
        return [[ ca, 0, sa],
                [  0, 1,  0],
                [-sa, 0, ca]]
    
    @staticmethod
    def Rz(a: float) -> List[List[float]]:
        ca, sa = math.cos(a), math.sin(a)
        return [[ca, -sa, 0],
                [sa,  ca, 0],
                [ 0,   0, 1]]



    
    # --------- Basic Matrix functions ----------

    @staticmethod
    def matmul3(A, B):
        # 3x3 matrix * 3x3 matrix
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    
    @staticmethod
    def matvec3(R, v):
        # 3x3 matrix * 3x1 vector
        return [sum(R[i][k]*v[k] for k in range(3)) for i in range(3)]

    @staticmethod
    def vec_add(a, b):
        # 3D vect + 3D vect
        return [a[i] + b[i] for i in range(3)]
    
    @staticmethod
    def vec_sub(a, b):
        # 3D vect - 3D vect
        return [a[i] - b[i] for i in range(3)]
    
    @staticmethod
    def RT(R):
        # Transpose of 3x3 matrix
        return [[R[j][i] for j in range(3)] for i in range(3)]

    @staticmethod
    def skew(w):
            return np.array([
                [0,     -w[2],  w[1]],
                [w[2],   0,    -w[0]],
                [-w[1],  w[0],  0]
            ])




    # ---------- Getting R and t from 6 coords ---------------

    def make_rot_trans(tx: float, ty: float, tz: float,
                    rx: float, ry: float, rz: float) -> Tuple[List[List[float]], List[float]]:

        Rx_m = LinAlgeb.Rx(rx)
        Ry_m = LinAlgeb.Ry(ry)
        Rz_m = LinAlgeb.Rz(rz)

        # R = Rz * Ry * Rx multiplying them give us how the rotation matrix expressing how the camera is rotated around its state before rotating
        R = LinAlgeb.matmul3(Rz_m, LinAlgeb.matmul3(Ry_m, Rx_m))
        t = [tx, ty, tz]
        return R, t


    # ------------- Getting homog matrix from R and t --------------

    @staticmethod
    def get_homog_matrix(R: Sequence[Sequence[float]], t: Sequence[float]) -> np.ndarray:
        """Build 4x4 homogeneous transformation matrix from R and t."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T



    # ------------- Direct transform frame of a point using homog matrix -------------------

    @staticmethod
    def transform_points_to_world(p_in_cam: List[Sequence[float]],
                        cam_homog : np.ndarray) -> List[float]:
        
        # (Camera -> World)
        # p_world = cur_cam_rot * p_cam + cur_cam_trans
        cam_rot = cam_homog[:3, :3]
        cam_trans = cam_homog[:3, 3]

        points_c = []
        for p in p_in_cam:
            Rp = LinAlgeb.matvec3(cam_rot, p)
            points_c.append(LinAlgeb.vec_add(Rp, cam_trans))
        return np.array(points_c) 


    # ------------ Inverse transform frame of a point using homog matrix --------------------  
     
    @staticmethod 
    def transform_points_to_cam(p_in_world: List[Sequence[float]],
                        cam_homog : np.ndarray) -> List[float]:

        # (in World -> in Camera)
        # p_cam = cam_rot^T * (p_world - cam_trans)
        cam_rot = cam_homog[:3, :3]
        cam_trans = cam_homog[:3, 3]
        points_c = []
        for p in p_in_world:
            diff = LinAlgeb.vec_sub(p, cam_trans)
            points_c.append(LinAlgeb.matvec3(LinAlgeb.RT(cam_rot), diff))
        return np.array(points_c)    




    @staticmethod
    def print_mat_condition_number(A):

        A = np.asarray(A)

        # ---- Fix IBVS stacked format (n,2,6) -> (2n,6)
        if A.ndim == 3:
            if A.shape[1] == 2:
                A = A.reshape(-1, A.shape[-1])
            else:
                raise ValueError("Unsupported 3D matrix shape")

        # ---- Ensure 2D matrix
        if A.ndim != 2:
            raise ValueError("Input must be a 2D matrix")

        # ---- Compute condition number
        cond_number = np.linalg.cond(A)

        # ---- Interpret result
        if np.isinf(cond_number):
            meaning = "Singular (rank deficient)"

        elif cond_number < 10:
            meaning = "Excellent conditioning"

        elif cond_number < 100:
            meaning = "Good conditioning"

        elif cond_number < 1e3:
            meaning = "Acceptable"

        elif cond_number < 1e5:
            meaning = "Unstable"

        elif cond_number < 1e7:
            meaning = "Very unstable"

        elif cond_number < 1e8:
            meaning = "Severely ill-conditioned"

        else:
            meaning = "Almost linearly dependent"

        print(f"{meaning}, {cond_number}")

