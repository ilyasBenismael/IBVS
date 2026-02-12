import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


class LiveOptimizationVisualizer:

    def __init__(self, initial_img):
        self.initial_img = initial_img
        
        # Storage for history
        self.iterations = []
        self.losses = []
        self.velocities = []  # List of [vx, vy, vz, wx, vy, wz]
               
        # Setup figure with adjusted layout
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
        
        # Top row: velocity plot (left) and loss plot (right)
        self.ax_vel = self.fig.add_subplot(gs[0, 0])
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        
        # Bottom row: diff image (left) and comparison grid (right)
        self.ax_diff= self.fig.add_subplot(gs[1, 0])
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

        # ===== diff =====
        self.ax_diff.set_title('Diff')
        self.ax_diff.axis('off')
        self.diff_img_plot = None

        # ===== Nested GridSpec, the 4 subbplot inside the big 4 plots =====
        parent_spec = self.ax_comparison.get_subplotspec()
        self.ax_comparison.remove()

        gs_inner = parent_spec.subgridspec(2, 2, wspace=0.02, hspace=0.15)

        self.ax_img_cur = self.fig.add_subplot(gs_inner[0, 0])
        self.ax_img_des = self.fig.add_subplot(gs_inner[0, 1])
        self.ax_img_real = self.fig.add_subplot(gs_inner[1, 0])
        self.ax_img_init = self.fig.add_subplot(gs_inner[1, 1])

        for ax in (self.ax_img_cur, self.ax_img_des, self.ax_img_real, self.ax_img_init):
            ax.axis('off')

        self.ax_img_cur.set_title("Current")
        self.ax_img_des.set_title("Desired")
        self.ax_img_real.set_title("Real")
        self.ax_img_init.set_title("Initial")

        self.img_cur_plot = None
        self.img_des_plot = None
        self.img_real_plot = None
        self.img_init_plot = None
        




#i will have current , desired, initial, real
    def update(self, iteration, diff_img, current_img, real_img, des_img, velocity, loss):

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

        # ----- diff -----
        if self.diff_img_plot is None:
            self.diff_img_plot = self.ax_diff.imshow(diff_img)
        else:
            self.diff_img_plot.set_data(diff_img)


        # ----- images -----
        if self.img_cur_plot is None:
            self.img_cur_plot = self.ax_img_cur.imshow(current_img)
            self.img_des_plot = self.ax_img_des.imshow(des_img)
            self.img_real_plot = self.ax_img_real.imshow(real_img)
            self.img_init_plot = self.ax_img_init.imshow(self.initial_img)
        else:
            
            self.img_cur_plot.set_data(current_img)
            self.img_des_plot.set_data(des_img)
            self.img_real_plot.set_data(real_img)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()




    def close(self):
        plt.ioff()
        plt.close(self.fig)







