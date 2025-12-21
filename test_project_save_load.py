"""
项目保存和加载测试脚本
用于测试项目保存和加载功能，包含详细的调试信息
"""
import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMessageBox
from src.ui.main_window import SpectraConfigDialog
from src.core.project_save_manager import ProjectSaveManager


def print_debug(msg, level="INFO"):
    """打印调试信息"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {msg}")


def test_project_save_load():
    """测试项目保存和加载"""
    print_debug("=" * 80)
    print_debug("开始项目保存和加载测试")
    print_debug("=" * 80)
    
    app = QApplication(sys.argv)
    
    try:
        # 步骤1: 创建主窗口
        print_debug("步骤1: 创建主窗口...")
        try:
            window = SpectraConfigDialog()
            print_debug("主窗口创建成功")
            print_debug(f"主布局子项数量: {window.main_layout.count()}")
            print_debug(f"是否有菜单栏: {hasattr(window, 'menu_bar')}")
            if hasattr(window, 'menu_bar'):
                print_debug(f"菜单栏存在: {window.menu_bar is not None}")
        except Exception as e:
            print_debug(f"创建主窗口失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return False
        
        # 步骤2: 设置一些测试数据
        print_debug("步骤2: 设置测试数据...")
        if hasattr(window, 'folder_input'):
            # 设置一个测试文件夹路径（如果存在）
            test_folder = project_root / "test_data"
            if test_folder.exists():
                window.folder_input.setText(str(test_folder))
                print_debug(f"设置测试文件夹: {test_folder}")
            else:
                print_debug(f"测试文件夹不存在: {test_folder}，跳过")
        
        # 步骤3: 保存项目
        print_debug("步骤3: 保存项目...")
        project_save_manager = ProjectSaveManager()
        test_project_path = project_save_manager.projects_dir / f"test_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        print_debug(f"保存路径: {test_project_path}")
        success = project_save_manager.save_project(str(test_project_path), window, note="测试项目")
        
        if not success:
            print_debug("保存项目失败！", "ERROR")
            return False
        
        print_debug("项目保存成功")
        
        # 验证文件是否存在
        if not test_project_path.exists():
            print_debug("保存的文件不存在！", "ERROR")
            return False
        
        print_debug(f"文件已创建: {test_project_path}")
        print_debug(f"文件大小: {test_project_path.stat().st_size} 字节")
        
        # 步骤4: 读取并验证保存的数据
        print_debug("步骤4: 验证保存的数据...")
        try:
            with open(test_project_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            print_debug(f"保存的数据键: {list(saved_data.keys())}")
            
            if 'csv_folder_path' in saved_data:
                print_debug(f"CSV路径: {saved_data['csv_folder_path']}")
            
            if 'plot_config' in saved_data:
                print_debug(f"PlotConfig键: {list(saved_data['plot_config'].keys())}")
            
            if 'data_states' in saved_data:
                print_debug(f"数据状态键: {list(saved_data['data_states'].keys())}")
                if 'plot_windows' in saved_data['data_states']:
                    plot_windows = saved_data['data_states']['plot_windows']
                    print_debug(f"绘图窗口数量: {len(plot_windows)}")
                    for name, data in plot_windows.items():
                        print_debug(f"  窗口 '{name}': 可见={data.get('is_visible', False)}, 有绘图数据={bool(data.get('plot_data'))}")
            
            if 'other_info' in saved_data:
                print_debug(f"其他信息键: {list(saved_data['other_info'].keys())}")
        except Exception as e:
            print_debug(f"验证保存数据失败: {e}", "ERROR")
            traceback.print_exc()
            return False
        
        # 步骤5: 创建新窗口并加载项目
        print_debug("步骤5: 创建新窗口并加载项目...")
        window2 = SpectraConfigDialog()
        print_debug("新窗口创建成功")
        
        # 步骤6: 加载项目
        print_debug("步骤6: 加载项目...")
        print_debug(f"加载路径: {test_project_path}")
        
        success = project_save_manager.load_project(str(test_project_path), window2)
        
        if not success:
            print_debug("加载项目失败！", "ERROR")
            return False
        
        print_debug("项目加载成功")
        
        # 步骤7: 验证加载的数据
        print_debug("步骤7: 验证加载的数据...")
        
        if hasattr(window2, 'folder_input'):
            loaded_folder = window2.folder_input.text()
            print_debug(f"加载的CSV路径: {loaded_folder}")
        
        if hasattr(window2, 'plot_windows'):
            print_debug(f"绘图窗口数量: {len(window2.plot_windows)}")
            for name, win in window2.plot_windows.items():
                if win:
                    print_debug(f"  窗口 '{name}': 可见={win.isVisible()}, 有绘图数据={bool(hasattr(win, 'current_plot_data') and win.current_plot_data)}")
        
        # 步骤8: 显示窗口（非阻塞）
        print_debug("步骤8: 显示窗口...")
        window2.show()
        print_debug("窗口已显示")
        
        # 等待一段时间让窗口完全初始化
        print_debug("等待窗口初始化...")
        from PyQt6.QtCore import QTimer
        timer = QTimer()
        timer.setSingleShot(True)
        
        def check_ui():
            print_debug("检查UI状态...")
            try:
                # 检查主窗口是否可见
                if window2.isVisible():
                    print_debug("主窗口可见")
                else:
                    print_debug("主窗口不可见", "WARNING")
                
                # 检查是否有布局
                if hasattr(window2, 'main_layout'):
                    print_debug(f"主布局存在，子项数量: {window2.main_layout.count()}")
                else:
                    print_debug("主布局不存在", "WARNING")
                
                # 检查是否有菜单栏
                if hasattr(window2, 'menu_bar'):
                    print_debug(f"菜单栏存在")
                else:
                    print_debug("菜单栏不存在", "WARNING")
                
                print_debug("=" * 80)
                print_debug("测试完成！窗口将保持打开状态5秒")
                print_debug("=" * 80)
                
                # 5秒后关闭
                QTimer.singleShot(5000, app.quit)
            except Exception as e:
                print_debug(f"检查UI状态失败: {e}", "ERROR")
                traceback.print_exc()
                app.quit()
        
        timer.timeout.connect(check_ui)
        timer.start(1000)  # 1秒后检查
        
        # 运行事件循环
        print_debug("运行事件循环...")
        sys.exit(app.exec())
        
    except Exception as e:
        print_debug(f"测试过程中发生错误: {e}", "ERROR")
        traceback.print_exc()
        QMessageBox.critical(None, "测试错误", f"测试失败:\n{str(e)}\n\n详细错误:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    test_project_save_load()

