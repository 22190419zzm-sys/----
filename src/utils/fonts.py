import os
from matplotlib.font_manager import FontProperties, findSystemFonts
import matplotlib.pyplot as plt


def setup_matplotlib_fonts():
    # 尝试查找 Times New Roman
    try:
        times_path = next((f for f in findSystemFonts(fontext='ttf') if 'times' in os.path.basename(f).lower()), None)
        if times_path:
            plt.rcParams['font.serif'] = [FontProperties(fname=times_path).get_name(), 'Times New Roman', 'serif']
        else:
            plt.rcParams['font.serif'] = ['Times New Roman', 'serif']
    except Exception:
        plt.rcParams['font.serif'] = ['Times New Roman', 'serif']

    # 尝试查找 SimHei 用于中文
    try:
        font_files = findSystemFonts(fontpaths=None, fontext='ttf')
        simhei_path = next((f for f in font_files if 'simhei' in os.path.basename(f).lower()), None)
        if simhei_path:
            # FIX: 修正 zh_font 未定义的问题
            zh_font = FontProperties(fname=simhei_path)
            plt.rcParams['font.sans-serif'] = [zh_font.get_name(), 'Arial Unicode MS']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    except Exception:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

