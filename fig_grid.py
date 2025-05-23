import wavefunction as wf
import grids as gr
import plot_grid as plot
from PIL import Image
import os

def get_k(n):
    n = abs(n)
    return [x for x in range(-n, n) if x != 0]

def get_m(n):
    n = abs(n)
    return [x for x in [i + 0.5 for i in range(-n, n)] if x != 0]

def get_n(k):
    if k > 0:
        return k + 1
    else:
        return abs(k)

def get_grid_params(k):
    points_per_dim = 300

    range_sizes = {
        1: 10,
        2: 15,
        3: 25,
        4: 40,
        5: 60,
        6: 70,
    }

    n = k + 1 if k > 0 else -k
    range_size = range_sizes.get(n, 0)
    
    return range_size, points_per_dim

k_values = get_k(6)
m_values = get_m(6)


output_dir = "dirac_grid"
os.makedirs(output_dir, exist_ok=True)


plt_config = plot.DiracPlotConfig(
    density_scale=0.1,
    psi_scale=0.1,
    points_per_unit=14,
    use_fixed_points=True,
    width=1500,
    height=1000,
    camera=dict(eye=dict(x=1.3, y=1.3, z=1.3), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0))
)


# 生成并保存每个子图
for i, k in enumerate(k_values):
    n = get_n(k)
    valid_m = get_m(k)
    # 获取动态网格参数
    range_size, points_per_dim = get_grid_params(k)
    grid = gr.GridGenerator(range_size, points_per_dim)
    X, Y, Z = grid.generate_grid()
    
    # 绘图配置
    
    for j, m in enumerate(m_values):
        if m not in valid_m:
            # 为无效组合生成空白图片
            blank_img = Image.new('RGB', (1500, 1000), color='white')
            blank_img.save(os.path.join(output_dir, f'k_{k}_m_{m:.1f}.png'))
            continue
        try:
            # 计算 Dirac 波函数
            Psi_dirac = wf.DiracHydrogen(n, k, m, 1)
            print(f"计算 k={k}, m={m}, n={n}, n_k={Psi_dirac.n_k}, range_size={range_size}, points_per_dim={points_per_dim}")
            spinor = Psi_dirac.compute_psi_xyz(X, Y, Z, t=0)
            
            # 生成图
            fig = plot.Dirac_plot(spinor, grid, plt_config)
            
            # 保存为 PNG
            output_path = os.path.join(output_dir, f'k_{k}_m_{m:.1f}.png')
            fig.write_image(output_path, format='png')
            print(f"已保存子图: {output_path}")
            
        except Exception as e:
            print(f"错误 k={k}, m={m}, n={n}: {e}")
            # 为错误情况生成占位图
            error_img = Image.new('RGB', (1500, 1000), color='white')
            error_img.save(os.path.join(output_dir, f'k_{k}_m_{m:.1f}.png'))
            continue

rows, cols = len(k_values), len(m_values)
subplot_width, subplot_height = 1500, 1000
grid_img = Image.new('RGB', (cols * subplot_width, rows * subplot_height), color='white')

# 添加图片到网格
for i, k in enumerate(k_values):
    for j, m in enumerate(m_values):
        img_path = os.path.join(output_dir, f'k_{k}_m_{m:.1f}.png')
        try:
            img = Image.open(img_path)
            # 确保图片大小一致
            img = img.resize((subplot_width, subplot_height), Image.LANCZOS)

            paste_y = (rows - 1 - i) * subplot_height
            paste_x = j * subplot_width
            
            grid_img.paste(img, (paste_x, paste_y))
        except Exception as e:
            print(f"读取图片错误 {img_path}: {e}")
            # 使用空白图片
            blank_img = Image.new('RGB', (subplot_width, subplot_height), color='white')
            grid_img.paste(blank_img, (j * subplot_width, i * subplot_height))

# 保存最终拼接图片
grid_img.save("dirac_wavefunction_grid.png")
print("已保存拼接网格图片: dirac_wavefunction_grid.png")
