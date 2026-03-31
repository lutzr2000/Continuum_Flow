import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np 
from Obstacles import circle

from netCDF4 import Dataset
import os
import imageio
from Obstacles import circle

def plot_all_to_mp4(nc_path, obstacle_mask, outpath, dt, video_name="simulation.mp4"):
    os.makedirs(outpath, exist_ok=True)
    
    dataset = Dataset(nc_path, 'r')

    X = np.array(dataset.variables['x'][:])
    Y = np.array(dataset.variables['y'][:])
    X, Y = np.meshgrid(X, Y)

    u_all = np.array(dataset.variables['u'][:])
    v_all = np.array(dataset.variables['v'][:])
    p_all = np.array(dataset.variables['p'][:])

    nt = u_all.shape[0]

    # ---------------- Figure & Layout ----------------
    font_size = 20
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
    })

    fig, axes = plt.subplots(2, 1, figsize=(19.2, 10.8), dpi=100, constrained_layout=True)

    # ---------------- Initial plots ----------------
    speed0 = np.sqrt(u_all[0]**2 + v_all[0]**2)
    speed0[obstacle_mask] = np.nan
    cmap_speed = cm.viridis.copy()
    cmap_speed.set_bad(color='black')
    im_v = axes[0].imshow(speed0, origin='lower', cmap=cmap_speed, vmin=0, vmax=12)
    cbar_v = fig.colorbar(im_v, ax=axes[0], orientation='vertical')
    cbar_v.set_label('Velocity Magnitude [m/s]', fontsize=font_size)
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].axis('equal')

    p0 = p_all[0].copy()
    p0[obstacle_mask] = np.nan
    cmap_p = cm.coolwarm.copy()
    cmap_p.set_bad(color='black')
    im_p = axes[1].imshow(p0, origin='lower', cmap=cmap_p, vmin=-50, vmax=50)
    cbar_p = fig.colorbar(im_p, ax=axes[1], orientation='vertical')
    cbar_p.set_label('Pressure [Pa]', fontsize=font_size)
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    axes[1].axis('equal')

    for spine in axes[0].spines.values():
        spine.set_linewidth(2.5)
    for spine in axes[1].spines.values():
        spine.set_linewidth(2.5)

    # ---------------- Video Writer ----------------
    video_path = os.path.join(outpath, video_name)
    writer = imageio.get_writer(video_path, fps=int(1/dt))

    for n in range(nt):
        print(f"Rendering frame {n+1}/{nt}...")

        # Update velocity
        speed = np.sqrt(u_all[n]**2 + v_all[n]**2)
        speed[obstacle_mask] = np.nan
        im_v.set_data(speed)
        axes[0].set_title(f'Velocity Magnitude at Time {n*dt:.4f} s')

        # Update pressure
        p_frame = p_all[n].copy()
        p_frame[obstacle_mask] = np.nan
        im_p.set_data(p_frame)
        axes[1].set_title(f'Pressure Field at Time {n*dt:.4f} s')

        # Canvas → RGB
        buffer, (w, h) = fig.canvas.print_to_buffer()
        image = np.frombuffer(buffer, dtype=np.uint8).reshape((h, w, 4))
        image = image[:, :, :3]  # RGB

        # Write frame to video
        writer.append_data(image)

    writer.close()
    plt.close(fig)
    dataset.close()
    print(f"Video saved to {video_path}")

# resolution
dt=1/24
delta = 0.04
nx = 1024
ny = 128
x = np.linspace(0,(nx-1)*delta,nx)
y = np.linspace(0,(ny-1)*delta,ny)
X, Y = np.meshgrid(x, y)


# Geometry
circle1_mask = circle(X,Y,nx*0.2*delta,ny*0.5*delta,0.5)
circle2_mask = circle(X,Y,nx*0.25*delta,ny*0.3*delta,0.6)
circle3_mask = circle(X,Y,nx*0.4*delta,ny*0.8*delta,0.3)

obstacle_mask = circle1_mask | circle2_mask | circle3_mask

nc_path = r"C:\Blenderzeug\BlenderCFD\Test\Test.nc"
outpath = r"C:\Blenderzeug\BlenderCFD\Test"

plot_all_to_mp4(nc_path, obstacle_mask, outpath, dt)