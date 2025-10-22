"""
诊断matplotlib显示问题
"""

import sys
import os

print("=" * 60)
print("Matplotlib 显示环境诊断")
print("=" * 60)

# 1. 检查matplotlib安装
print("\n1. 检查matplotlib安装...")
try:
    import matplotlib
    print(f"✓ matplotlib version: {matplotlib.__version__}")
except ImportError:
    print("✗ matplotlib未安装")
    print("  安装: pip install matplotlib")
    sys.exit(1)

# 2. 检查当前后端
print("\n2. 检查当前后端...")
print(f"   默认后端: {matplotlib.get_backend()}")

# 3. 列出所有可用后端
print("\n3. 可用的后端:")
from matplotlib import pyplot as plt
backends = [
    'TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'GTK4Agg',
    'WXAgg', 'MacOSX', 'Agg'
]

available_backends = []
for backend in backends:
    try:
        matplotlib.use(backend, force=True)
        available_backends.append(backend)
        print(f"   ✓ {backend}")
    except:
        print(f"   ✗ {backend}")

if not available_backends:
    print("\n⚠ 警告: 没有可用的交互式后端!")

# 4. 检查DISPLAY环境变量
print("\n4. 检查DISPLAY环境变量...")
display = os.environ.get('DISPLAY', None)
if display:
    print(f"   ✓ DISPLAY={display}")
else:
    print("   ✗ DISPLAY未设置 (无法显示GUI窗口)")
    print("   解决方案:")
    print("   - 本地运行: 确保有图形界面")
    print("   - SSH远程: 使用 ssh -X 启用X11转发")
    print("   - 或使用 game_3d_file.py (保存到文件)")

# 5. 测试简单绘图
print("\n5. 测试简单绘图...")
try:
    # 尝试使用最佳后端
    if 'TkAgg' in available_backends:
        matplotlib.use('TkAgg', force=True)
        print("   使用后端: TkAgg")
    elif 'Qt5Agg' in available_backends:
        matplotlib.use('Qt5Agg', force=True)
        print("   使用后端: Qt5Agg")
    else:
        matplotlib.use('Agg', force=True)
        print("   使用后端: Agg (非交互式)")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制简单立方体
    r = [0, 1]
    X, Y = np.meshgrid(r, r)
    Z = np.zeros_like(X)

    ax.plot_surface(X, Y, Z, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Test 3D Plot')

    # 尝试显示
    if matplotlib.get_backend() != 'Agg':
        print("   尝试显示窗口...")
        plt.ion()
        plt.show()
        plt.pause(0.1)
        print("   ✓ 如果看到窗口,说明可以正常使用game_3d.py")
        print("   ✗ 如果没有窗口,请使用game_3d_file.py")
    else:
        # 保存到文件
        output_file = "/tmp/matplotlib_test.png"
        plt.savefig(output_file)
        print(f"   ✓ 测试图片已保存: {output_file}")
        print("   建议使用: game_3d_file.py (保存到文件模式)")

    plt.close(fig)

except Exception as e:
    print(f"   ✗ 测试失败: {e}")

# 6. 推荐方案
print("\n" + "=" * 60)
print("推荐方案:")
print("=" * 60)

if display and available_backends and matplotlib.get_backend() != 'Agg':
    print("""
✓ 您的环境支持GUI显示

推荐使用:
  python game_3d.py

如果窗口不显示,尝试:
  1. 安装tkinter: sudo apt-get install python3-tk
  2. 使用SSH时启用X11转发: ssh -X user@host
""")
else:
    print("""
⚠ 您的环境可能不支持GUI显示

推荐使用:
  python game_3d_file.py

这会将所有可视化保存为PNG图片,功能完全相同!

示例:
  $ python game_3d_file.py
  > load 2x2x2 puzzle_001 42
  ✓ 已保存到: /tmp/puzzle_initial_0_143025.png

  > place 0
  Mode: rot
  Position: 1 1 1
  Rotation: 0
  ✓ 已保存到: /tmp/puzzle_placed_0_1_143056.png

然后查看图片:
  $ ls /tmp/puzzle_*.png
  $ open /tmp/puzzle_*.png  # macOS
  $ xdg-open /tmp/puzzle_*.png  # Linux
""")

print("\n" + "=" * 60)
