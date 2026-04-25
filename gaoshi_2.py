import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from scipy import integrate
import trimesh
from libmusic import rluxgo, initialize_music, muon_transport

def modified_gaisser(E_mu, cos_theta):
    P1, P2, P3, P4, P5 = 0.102573, -0.068287, 0.958633, 0.0407253, 0.817285
    cos_theta_star = np.sqrt(
        (cos_theta**2 + P1**2 + P2*cos_theta**P3 + P4*cos_theta**P5)
        / (1 + P1**2 + P2 + P4)
    )
    E_mu_star = E_mu * (1 + 3.64 / (E_mu * cos_theta_star**1.29))
    term1 = 1 / (1 + 1.1 * E_mu * cos_theta_star / 115)
    term2 = 0.054 / (1 + 1.1 * E_mu * cos_theta_star / 850)
    return 0.14 * (E_mu_star ** (-2.7)) * (term1 + term2)

def energy_theoretical(E):
    return integrate.quad(lambda ct: modified_gaisser(E, ct),0.342, 1)[0]

def cos_theoretical(ct):
    return integrate.quad(lambda E: modified_gaisser(E, ct), 1, 1000)[0]

def generate_muon_samples(num_samples=1000000, E_min=1e2, E_max=1e4, theta_bins=1000, E_bins=1000):
    #  构建网格
    E_edges = np.linspace(E_min, E_max, E_bins + 1)
    cos_theta_edges = np.linspace(0.342, 1.0, theta_bins + 1)
    
    E_centers = (E_edges[:-1] + E_edges[1:]) / 2      # (E_bins,)
    cos_theta_centers = (cos_theta_edges[:-1] + cos_theta_edges[1:]) / 2  # (theta_bins,)
    
    # 计算通量网格 —— 确保 modified_gaisser 支持广播！
    # 如果 modified_gaisser 不支持向量化，先用 np.vectorize 包装一次（只做一次）
    if not hasattr(modified_gaisser, '__vectorized__'):
        # 只在第一次调用时包装（可选）
        flux_func = np.vectorize(modified_gaisser)
    else:
        flux_func = modified_gaisser

    E_grid = E_centers[:, None]          # shape: (E_bins, 1)
    theta_grid = cos_theta_centers[None, :]  # shape: (1, theta_bins)
    flux = flux_func(E_grid, theta_grid)     # shape: (E_bins, theta_bins)

    #扁平化并归一化概率
    flux_flat = flux.ravel()             # (E_bins * theta_bins,)
    prob_flat = flux_flat / flux_flat.sum()
    cum_prob_flat = np.cumsum(prob_flat)

    # 采样索引
    u = np.random.rand(num_samples)
    idx = np.searchsorted(cum_prob_flat, u)  # shape: (num_samples,)

    # 向量化计算 bin 索引
    e_idx = idx // theta_bins            # (num_samples,)
    t_idx = idx % theta_bins             # (num_samples,)

    #  向量化均匀采样 
    # 在每个 bin 内随机采样（而不是取中心）
    E_low = E_edges[e_idx]
    E_high = E_edges[e_idx + 1]
    energies = np.random.uniform(E_low, E_high)

    cos_t_low = cos_theta_edges[t_idx]
    cos_t_high = cos_theta_edges[t_idx + 1]
    cos_thetas = np.random.uniform(cos_t_low, cos_t_high)

    return energies, cos_thetas


# 示例地形函数（替换成你的 DEM 插值函数）
def gaussian_mountain_anisotropic(x, y, H=300, sigma_x=80, sigma_y=120):
    """各向异性高斯山：z = H * exp(-x²/(2σx²) - y²/(2σy²))"""
    return H * np.exp(-(x**2 / (2 * sigma_x**2) + y**2 / (2 * sigma_y**2)))

def find_muon_entry_point(x_det, y_det, z_det, zenith, azimuth,
                          terrain_func,
                          r_max=50000.0,  # 最大追踪距离（米）
                          tol=0.1):
    """
    从探测器反向追踪缪子，找到其穿出地表的位置。
    
    参数:
        x_det, y_det, z_det: 探测器坐标（米）
        zenith: 天顶角（弧度），0 = 垂直向下
        azimuth: 方位角（弧度），从 x 轴（东）逆时针
        terrain_func: f(x, y) -> 地表高程 z
        r_max: 最大追踪距离（防止无限延伸）
        tol: 求解容差（米）
    
    返回:
        dict with 'x_entry', 'y_entry', 'z_entry', 'path_length'
        or None if no intersection found
    """
    # 反向方向向量（从探测器指向缪子来源）
    dx = -np.sin(zenith) * np.cos(azimuth)
    dy = -np.sin(zenith) * np.sin(azimuth)
    dz =  np.cos(zenith)   # 向上为正

    def height_diff(r):
        x = x_det + r * dx
        y = y_det + r * dy
        z_ray = z_det + r * dz
        z_surf = terrain_func(x, y)
        return z_ray - z_surf  # 当 ray 高于地表时 >0

    # 在 r=0 处，探测器应在地下：z_ray < z_surf → height_diff(0) < 0
    h0 = height_diff(0.0)
    h_max = height_diff(r_max)

    # 如果起点已在地表以上，不合理
    if (h0 >= 0).all():
        print("Warning: Detector is above or on surface!")
        return None

    # 如果最大距离处仍低于地表，可能方向不对或 r_max 太小
    if (h_max < 0).all():
        return None  # 未穿出地表（如水平射向山体内部）

    # 寻找根：height_diff(r) = 0
    try:
        sol = root_scalar(height_diff, bracket=[0.0, r_max], method='bisect', xtol=tol)
        if sol.converged:
            r = sol.root
            x_entry = x_det + r * dx
            y_entry = y_det + r * dy
            z_entry = z_det + r * dz  # 应 ≈ terrain_func(x_entry, y_entry)
            return {
                'x_entry': x_entry,
                'y_entry': y_entry,
                'z_entry': z_entry,
                'path_length': r,  # 从探测器到地表的距离
                'direction_vector': (dx, dy, dz)
            }
        else:
            return None
    except ValueError:
        return None
    
def compute_muon_path_through_mesh_cavity(
    detector_pos,
    zenith_angles,
    azimuth_angles,
    cavity_mesh,  # trimesh.Trimesh 对象
    t_max=2000.0,
    n_jobs=2    # 并行线程数
):
    """
    计算缪子穿过任意三角网格空腔的路径长度。
    
    参数:
        detector_pos: 探测器位置 (x, y, z)
        zenith_angles: 天顶角数组（弧度）
        azimuth_angles: 方位角数组（弧度）
        cavity_mesh: trimesh.Trimesh 表示的空腔表面（封闭）
        t_max: 最大追踪距离
    
    返回:
        path_lengths: 路径长度数组（米）
        has_intersection: 是否穿过空腔
    """
    detector_pos = np.asarray(detector_pos, dtype=float)
    N = len(zenith_angles)

    # 构建入射方向（从天空指向探测器）
    theta = np.asarray(zenith_angles)
    phi = np.asarray(azimuth_angles)
    directions = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        -np.cos(theta)
    ], axis=1)  # (N, 3)

    # 射线起点：从探测器反向延伸 t_max（确保覆盖整个空腔）
    ray_origins = detector_pos + directions * t_max  # 从远处开始
    ray_directions = -directions  # 指向探测器（即缪子飞行方向）

    # 使用 trimesh 批量求交（自动使用 Embree 加速）
    locations, index_ray, index_tri = cavity_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=True  # 获取所有交点
    )

    # 初始化结果
    path_lengths = np.zeros(N)
    has_intersect = np.zeros(N, dtype=bool)

    if len(locations) == 0:
        return path_lengths, has_intersect

    # 按射线索引分组
    from collections import defaultdict
    hits = defaultdict(list)
    for loc, ray_idx in zip(locations, index_ray):
        # 计算从射线起点到交点的距离
        dist = np.linalg.norm(loc - ray_origins[ray_idx])
        hits[ray_idx].append(dist)

    # 对每条射线：排序交点 → 成对计算路径段
    for ray_idx, dists in hits.items():
        dists = sorted(dists)
        total_length = 0.0
        # 交点成对出现：进入→离开→进入→离开...
        for i in range(0, len(dists) - 1, 2):
            enter = dists[i]
            exit_ = dists[i + 1]
            segment = exit_ - enter
            # 限制在 [0, t_max] 范围内
            segment = max(0, min(segment, t_max))
            total_length += segment
        
        if total_length > 1e-6:  # 避免数值误差
            path_lengths[ray_idx] = total_length
            has_intersect[ray_idx] = True
    
    return path_lengths , has_intersect

def sim(N=100000):
    x_d, y_d, z_d = 0,0 ,-300  # 探测器位置


    # 生成 N 个样本
    energies, cos_thetas = generate_muon_samples(num_samples=N)
    theta1 = np.arccos(cos_thetas)
    phi1 = np.random.uniform(0, 2 * np.pi, size=N)
    cavity_mesh = trimesh.load('/home/yangxin/music-sim/src/tests/python/konqiang/spherical_cavity_r100.stl')
    
    path_lengths , has_intersect = compute_muon_path_through_mesh_cavity(
        detector_pos=(x_d, y_d, z_d),
        zenith_angles=theta1,
        azimuth_angles=phi1,
        cavity_mesh=cavity_mesh
    )
    
    emu = []
    theta = []
    phi = []
    length_ls = []

    a = 0
    i = 0
    while a < 50000 and i < len(energies):  # 防止越界
        if i % 100 == 0:
            print(f"Processed {i} muons, collected {a} valid ones.")
        

        emu0 = energies[i]
        th0 = theta1[i]
        phi2 = phi1[i]
        length_cavity = path_lengths[i]
        
        entry = find_muon_entry_point(x_d, y_d, z_d, th0, phi2, gaussian_mountain_anisotropic)
        if entry is not None:
            x0, y0, z0 = entry['x_entry'], entry['y_entry'], entry['z_entry']
            length1 = entry['path_length']
            #h = 300/np.cos(th0)
            length = length1 - length_cavity
            
            #h = h - length_cavity
            
            result = muon_transport(x0, y0, z0, th0, phi2, emu0, length, 2.6, 0)
            if getattr(result, "emu", 0) > 0.1:
                emu.append(result.emu)
                theta.append(result.theta * 180 / np.pi)
                phi.append(result.phi * 180 / np.pi)
                length_ls.append(result.length)
                a += 1  # 只有这里才计数

        i += 1

    print(f"Final count: a = {a}")
     

    arrival_ratio = a / N * 100
    print(a) 
    print(f"输运完成！到达比例：{arrival_ratio:.2f}%")
    np.savez("gaoshi mountain+spherical_cavity_r100 5万.npz", theta=theta, phi=phi, length_ls=length_ls, emu=emu)

    # ================================
    # 绘制 emu 直方图（对数Y轴）
    # ================================
    fig, ax = plt.subplots(figsize=(10, 6))

    bins_emu = np.linspace(0, 7500, 150)
    counts_emu, bins_emu_edges, patches = ax.hist(emu, bins=bins_emu, histtype='step', color='blue', linewidth=1.5)

    # 对数 Y 轴
    ax.set_yscale('log')

    # 标题和标签
    ax.set_title('Muon Energy Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Energy (GeV)', fontsize=12)
    ax.set_ylabel('Entries (log scale)', fontsize=12)

    # 计算峰值和中值
    mode_bin = np.argmax(counts_emu)
    mode_value = (bins_emu_edges[mode_bin] + bins_emu_edges[mode_bin + 1]) / 2
    median_value = np.median(emu)

    # 添加统计信息框（包含峰值和中值）
    stats_emu = (f"Entries: {len(emu):d}\n"
                f"Mean: {np.mean(emu):.1f} GeV\n"
                f"Std Dev: {np.std(emu):.1f} GeV\n"
                f"Median: {median_value:.1f} GeV\n"
                f"Mode: {mode_value:.1f} GeV\n"
                f"Min: {np.min(emu):.1f} GeV\n"
                f"Max: {np.max(emu):.1f} GeV")

    ax.text(0.98, 0.98, stats_emu, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))

    # 标记峰值
    ax.axvline(mode_value, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mode: {mode_value:.1f} GeV')
    ax.axvline(median_value, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'Median: {median_value:.1f} GeV')

    # 添加图例
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ================================
    # 绘制 phi 直方图（线性Y轴）
    # ================================
    fig, ax = plt.subplots(figsize=(10, 6))
    bins_phi = np.linspace(0, 360, 73)  # 73个边界产生72个bin，更准确
    counts_phi, bins_phi_edges, patches = ax.hist(phi, bins=bins_phi, histtype='step', color='green', linewidth=1.5)

    # 标题和标签
    ax.set_title('Azimuth Angle (φ) Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('φ (degrees)', fontsize=12)
    ax.set_ylabel('Entries', fontsize=12)

    # 计算峰值和中值
    mode_bin = np.argmax(counts_phi)
    mode_value = (bins_phi_edges[mode_bin] + bins_phi_edges[mode_bin + 1]) / 2
    median_value = np.median(phi)

    # 添加统计信息框
    stats_phi = (f"Entries: {len(phi):d}\n"
                f"Mean: {np.mean(phi):.1f}°\n"
                f"Std Dev: {np.std(phi):.1f}°\n"
                f"Median: {median_value:.1f}°\n"
                f"Mode: {mode_value:.1f}°")

    

    ax.text(0.5, 0.25, stats_phi, transform=ax.transAxes,
        fontsize=10,
        horizontalalignment='center',    # 水平居中
        verticalalignment='center',      # 垂直居中（相对于 y=0.25）
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white", edgecolor="black", alpha=0.9))
    
    # 标记峰值和中值
    ax.axvline(mode_value, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(median_value, color='orange', linestyle=':', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show()

    # ================================
    # 绘制 theta 直方图（线性Y轴）
    # ================================
    fig, ax = plt.subplots(figsize=(10, 6))

    bins_theta = np.linspace(0, 120, 121)  # 121个边界产生120个bin
    counts_theta, bins_theta_edges, patches = ax.hist(theta, bins=bins_theta, histtype='step', color='purple', linewidth=1.5)

    # 标题和标签
    ax.set_title('Zenith Angle (θ) Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('θ (degrees)', fontsize=12)
    ax.set_ylabel('Entries', fontsize=12)

    # 计算峰值和中值
    mode_bin = np.argmax(counts_theta)
    mode_value = (bins_theta_edges[mode_bin] + bins_theta_edges[mode_bin + 1]) / 2
    median_value = np.median(theta)

    # 添加统计信息框
    stats_theta = (f"Entries: {len(theta):d}\n"
                f"Mean: {np.mean(theta):.2f}°\n"
                f"Std Dev: {np.std(theta):.2f}°\n"
                f"Median: {median_value:.2f}°\n"
                f"Mode: {mode_value:.2f}°")

    ax.text(0.98, 0.98, stats_theta, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))

    # 标记峰值和中值
    ax.axvline(mode_value, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(median_value, color='orange', linestyle=':', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show()

    # ================================
    # 3. muon径迹长度分布
    # ================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # 自动确定合适的bins
    if len(length_ls) > 0:
        min_length = max(0, np.min(length_ls) - 50)
        max_length = np.max(length_ls) + 50
        n_bins = min(50, max(20, len(length_ls) // 100))  # 根据数据量确定bin数
        bins_length = np.linspace(min_length, max_length, n_bins)
    else:
        bins_length = np.linspace(300, 1000, 50)

    counts_length, bins_length_edges, patches = ax.hist(length_ls, bins=bins_length, 
                                                    histtype='step', color='darkblue', 
                                                    linewidth=2, edgecolor='navy')

    ax.set_title('Muon Path Length Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Path Length (m)', fontsize=12)
    ax.set_ylabel('Entries', fontsize=12)

    # 计算峰值和中值
    if len(counts_length) > 0:
        mode_bin = np.argmax(counts_length)
        mode_value = (bins_length_edges[mode_bin] + bins_length_edges[mode_bin + 1]) / 2
    else:
        mode_value = np.nan

    median_value = np.median(length_ls) if len(length_ls) > 0 else np.nan

    # 添加统计信息框
    stats_length = (f"Entries: {len(length_ls):d}\n"
                    f"Mean: {np.mean(length_ls):.2f} m\n"
                    f"Std Dev: {np.std(length_ls):.2f} m\n"
                    f"Median: {median_value:.2f} m\n"
                    f"Mode: {mode_value:.2f} m\n"
                    f"Min: {np.min(length_ls):.1f} m\n"
                    f"Max: {np.max(length_ls):.1f} m")

    ax.text(0.98, 0.98, stats_length, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))

    # 标记峰值和中值（如果数据有效）
    if not np.isnan(mode_value):
        ax.axvline(mode_value, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Mode: {mode_value:.1f} m')
    if not np.isnan(median_value):
        ax.axvline(median_value, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label=f'Median: {median_value:.1f} m')

    # 添加图例（如果有有效数据）
    if not (np.isnan(mode_value) and np.isnan(median_value)):
        ax.legend(loc='upper left', fontsize=9)

    # 添加网格线
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    # 穿过山体后的能量与theta或径迹长度的二维分布图
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # 第一个子图：能量与theta
    im1 = axs[0].hist2d(theta, emu, bins=50, cmap="plasma", alpha=0.8)
    axs[0].set_xlabel(r"Zenith Angle $\theta$ (°)", fontsize=12, fontweight='bold')
    axs[0].set_ylabel("Energy (GeV)", fontsize=12, fontweight='bold')
    axs[0].set_title("Energy vs. Zenith Angle", fontsize=14, fontweight='bold')
    axs[0].grid(True, alpha=0.3)
    plt.colorbar(im1[3], ax=axs[0], label='Counts')

    # 第二个子图：能量与径迹长度
    im2 = axs[1].hist2d(length_ls, emu, bins=50, cmap="viridis", alpha=0.8)
    axs[1].set_xlabel("Track Length (m)", fontsize=12, fontweight='bold')
    axs[1].set_ylabel("Energy (GeV)", fontsize=12, fontweight='bold')
    axs[1].set_title("Energy vs. Track Length", fontsize=14, fontweight='bold')
    axs[1].grid(True, alpha=0.3)
    plt.colorbar(im2[3], ax=axs[1], label='Counts')



    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    rluxgo(42)
    initialize_music()
    a = sim(N=1000000)

