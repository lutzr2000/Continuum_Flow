import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np 

def plot_velocity_pressure(X, Y, u, v, p, n, dt, obstacle_mask, outpath=rf"C:\Blenderzeug\BlenderCFD\Output"):
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

    # ---------------- Velocity ----------------
    speed = np.sqrt(u**2 + v**2)
    speed_colored = speed.copy()
    speed_colored[obstacle_mask] = np.nan  
    cmap = cm.viridis.copy()
    cmap.set_bad(color='black') 
    masked_speed = np.ma.masked_invalid(speed_colored)
    levels_v = np.linspace(0, 10, 200)
    cf_v = axes[0].contourf(X, Y, masked_speed, levels=levels_v, cmap=cmap)
    cbar_v = fig.colorbar(cf_v, ax=axes[0], orientation='vertical')
    cbar_v.set_label('Velocity Magnitude [m/s]', fontsize=font_size)

    # ---------------- Pressure ----------------
    p_colored = p.copy()
    p_colored[obstacle_mask] = np.nan
    cmap_p = cm.coolwarm.copy()
    cmap_p.set_bad(color='black')
    masked_p = np.ma.masked_invalid(p_colored)

    levels_p = np.linspace(-50, 50, 200)
    cf_p = axes[1].contourf(X, Y, masked_p, levels=levels_p, cmap=cmap_p)
    cbar_p = fig.colorbar(cf_p, ax=axes[1], orientation='vertical')
    cbar_p.set_label('Pressure [Pa]', fontsize=font_size)

    # ---------------- Layout ----------------
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].set_title(f'Velocity Magnitude at Time {np.round(n*dt,6)} seconds')
    axes[0].axis('equal')
    for spine in axes[0].spines.values():
        spine.set_linewidth(2.5)

    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    axes[1].set_title(f'Pressure Field at Time {np.round(n*dt,6)} seconds')
    axes[1].axis('equal')
    for spine in axes[1].spines.values():
        spine.set_linewidth(2.5)

    plt.savefig(outpath + rf"\step_{n}.png", dpi=100)
    plt.close()