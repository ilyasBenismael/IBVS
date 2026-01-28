import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


class LiveOptimizationVisualizer:

    def __init__(self, desired_img, initial_img):
        self.desired_img = desired_img
        self.initial_img = initial_img
        
        # Storage for history
        self.iterations = []
        self.losses = []
        self.velocities = []  # List of [vx, vy, vz, wx, vy, wz]
        
        # Compute initial difference
        self.initial_diff = self.compute_grayscale_difference(desired_img, initial_img)
        
        # Setup figure with adjusted layout
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
        
        # Top row: velocity plot (left) and loss plot (right)
        self.ax_vel = self.fig.add_subplot(gs[0, 0])
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        
        # Bottom row: scene image (left) and comparison grid (right)
        self.ax_scene = self.fig.add_subplot(gs[1, 0])
        self.ax_comparison = self.fig.add_subplot(gs[1, 1])
        
        # Initialize plots
        self._setup_plots()
        
        # Blitting setup
        plt.ion()
        plt.show()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        
        # Update frequency
        self.update_counter = 0
        self.plot_update_frequency = 1  # Update every N iterations
        
        # Patches to be created after first image
        self.rect_scene = None
        self.rect_desired = None
        self.rect_current = None
        self.rect_initial_diff = None
        self.rect_current_diff = None


    def compute_grayscale_difference(self, img1, img2, normalize=True):
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




    def _setup_plots(self):
        # ===== Velocity =====
        self.ax_vel.set_title('Velocity Components Over Iterations')
        self.ax_vel.set_xlabel('Iteration')
        self.ax_vel.set_ylabel('Velocity')
        self.ax_vel.grid(True, alpha=0.3)

        self.vel_lines = []
        labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for label, color in zip(labels, colors):
            line, = self.ax_vel.plot([], [], label=label, color=color)
            self.vel_lines.append(line)
        self.ax_vel.legend()

        # ===== Loss =====
        self.ax_loss.set_title('Loss Over Iterations')
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        self.loss_line, = self.ax_loss.plot([], [], 'b-')

        # ===== Scene =====
        self.ax_scene.set_title('Current Frame')
        self.ax_scene.axis('off')
        self.scene_img_plot = None

        # ===== Replace ax_comparison with nested GridSpec =====
        parent_spec = self.ax_comparison.get_subplotspec()
        self.ax_comparison.remove()

        gs_inner = parent_spec.subgridspec(2, 2, wspace=0.02, hspace=0.15)

        self.ax_img_cur = self.fig.add_subplot(gs_inner[0, 0])
        self.ax_img_des = self.fig.add_subplot(gs_inner[0, 1])
        self.ax_img_init = self.fig.add_subplot(gs_inner[1, 0])
        self.ax_img_diff = self.fig.add_subplot(gs_inner[1, 1])

        for ax in (self.ax_img_cur, self.ax_img_des, self.ax_img_init, self.ax_img_diff):
            ax.axis('off')

        self.ax_img_cur.set_title("Current")
        self.ax_img_des.set_title("Desired")
        self.ax_img_init.set_title("Initial")
        self.ax_img_diff.set_title("Difference")

        self.img_cur_plot = None
        self.img_des_plot = None
        self.img_init_plot = None
        self.img_diff_plot = None





    def update(self, iteration, scene_img, current_img, diff_img, velocity, loss):

        self.iterations.append(iteration)
        self.losses.append(loss)
        self.velocities.append(velocity)

        # ----- plots -----
        vel_array = np.array(self.velocities)
        for i, line in enumerate(self.vel_lines):
            line.set_data(self.iterations, vel_array[:, i])
        self.ax_vel.relim()
        self.ax_vel.autoscale_view()

        self.loss_line.set_data(self.iterations, self.losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        # ----- scene -----
        if self.scene_img_plot is None:
            self.scene_img_plot = self.ax_scene.imshow(scene_img)
        else:
            self.scene_img_plot.set_data(scene_img)

        # ----- images -----
        if self.img_cur_plot is None:
            self.img_cur_plot = self.ax_img_cur.imshow(current_img)
            self.img_des_plot = self.ax_img_des.imshow(self.desired_img)
            self.img_init_plot = self.ax_img_init.imshow(self.initial_img)
            self.img_diff_plot = self.ax_img_diff.imshow(diff_img, cmap='gray')
        else:
            self.img_cur_plot.set_data(current_img)
            self.img_diff_plot.set_data(diff_img)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()





    def close(self):
        plt.ioff()
        plt.close(self.fig)







