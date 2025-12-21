import sys
import warnings
import traceback

# 过滤 PyQt6 的弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*sipPyTypeDict.*')

# 首先只导入最基础的PyQt6模块，确保启动画面能立即显示
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


def main():
    # 立即创建应用程序（不导入其他模块）
    app = QApplication(sys.argv)
    
    # 立即显示启动画面（使用最少的导入）
    # 直接创建启动画面窗口，不通过导入
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
    from PyQt6.QtGui import QFont
    
    class SplashScreen(QWidget):
        """启动画面窗口"""
        
        def __init__(self):
            super().__init__()
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint | 
                Qt.WindowType.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            
            # 设置窗口大小和位置
            self.setFixedSize(500, 200)
            self._center_window()
            
            # 创建布局
            layout = QVBoxLayout(self)
            layout.setSpacing(20)
            layout.setContentsMargins(30, 30, 30, 30)
            
            # 标题标签
            self.title_label = QLabel("白皓正在为你打开软件")
            title_font = QFont()
            title_font.setPointSize(16)
            title_font.setBold(True)
            self.title_label.setFont(title_font)
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.title_label.setStyleSheet("color: #2c3e50;")
            layout.addWidget(self.title_label)
            
            # 状态标签
            self.status_label = QLabel("正在初始化...")
            status_font = QFont()
            status_font.setPointSize(11)
            self.status_label.setFont(status_font)
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.status_label.setStyleSheet("color: #34495e;")
            layout.addWidget(self.status_label)
            
            # 进度条（只保留一个进度条，不再单独显示百分比标签）
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")  # 在进度条内部显示百分比文字
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #bdc3c7;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #ecf0f1;
                    height: 25px;
                }
                QProgressBar::chunk {
                    background-color: #3498db;
                    border-radius: 3px;
                }
            """)
            layout.addWidget(self.progress_bar)
            
            # 设置窗口样式
            self.setStyleSheet("""
                QWidget {
                    background-color: white;
                    border-radius: 10px;
                }
            """)
            
            # 加载步骤列表
            self.loading_steps = []
        
        def _center_window(self):
            """居中显示窗口"""
            screen = QApplication.primaryScreen().geometry()
            size = self.geometry()
            self.move(
                (screen.width() - size.width()) // 2,
                (screen.height() - size.height()) // 2
            )
        
        def set_loading_steps(self, steps):
            """设置加载步骤列表"""
            self.loading_steps = steps
        
        def update_progress(self, step_index, status_text=None):
            """更新进度"""
            if step_index < len(self.loading_steps):
                if status_text is None:
                    status_text = self.loading_steps[step_index]
                self.status_label.setText(status_text)
            
            # 计算进度百分比
            if len(self.loading_steps) > 0:
                progress = int((step_index + 1) / len(self.loading_steps) * 100)
                self.progress_bar.setValue(progress)
            
            # 强制更新界面
            QApplication.processEvents()
        
        def showEvent(self, event):
            """显示事件"""
            super().showEvent(event)
            self._center_window()
    
    # 立即创建并显示启动画面
    splash = SplashScreen()
    splash.show()
    
    # 强制处理事件以确保启动画面立即显示
    app.processEvents()
    
    # 现在开始加载模块（所有导入都在启动画面显示之后）
    import time
    import matplotlib
    matplotlib.use('QtAgg')
    
    # 定义加载步骤
    loading_steps = [
        "正在加载核心模块...",
        "正在加载数据处理模块...",
        "正在加载绘图模块...",
        "正在加载窗口模块...",
        "正在加载分析工具...",
        "正在初始化主窗口...",
        "正在加载配置...",
        "准备就绪！"
    ]
    
    splash.set_loading_steps(loading_steps)
    
    # 步骤1: 加载核心模块
    splash.update_progress(0, "正在加载核心模块...")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.optimize import nnls
    
    # 步骤2: 加载数据处理模块
    splash.update_progress(1, "正在加载数据处理模块...")
    from src.core.preprocessor import DataPreProcessor
    from src.utils.helpers import natural_sort_key, group_files_by_name
    
    # 步骤3: 加载绘图模块
    splash.update_progress(2, "正在加载绘图模块...")
    from src.ui.canvas import MplCanvas
    from src.ui.windows.plot_window import MplPlotWindow
    
    # 步骤4: 加载窗口模块
    splash.update_progress(3, "正在加载窗口模块...")
    from src.ui.windows.nmf_window import NMFResultWindow
    from src.ui.windows.quantitative_window import QuantitativeResultWindow, QuantitativeAnalysisDialog
    from src.ui.windows.nmf_validation_window import NMFFitValidationWindow
    from src.ui.windows.two_dcos_window import TwoDCOSWindow, TwoDCOSMarginalPlotWindow
    from src.ui.windows.classification_window import ClassificationResultWindow
    from src.ui.windows.dae_window import DAEComparisonWindow
    from src.ui.windows.batch_plot_window import BatchPlotWindow
    
    # 步骤5: 加载分析工具
    splash.update_progress(4, "正在加载分析工具...")
    from src.ui.controllers import DataController
    from src.ui.panels.nmf_panel import NMFPanelMixin
    from src.ui.panels.cos_panel import COSPanelMixin
    from src.ui.panels.classify_panel import ClassifyPanelMixin
    
    # 步骤6: 初始化主窗口
    splash.update_progress(5, "正在初始化主窗口...")
    try:
        from src.ui.main_window import SpectraConfigDialog
        window = SpectraConfigDialog()
    except Exception as e:
        splash.close()
        from PyQt6.QtWidgets import QMessageBox
        error_msg = f"初始化主窗口失败：\n{str(e)}\n\n详细错误信息：\n{traceback.format_exc()}"
        QMessageBox.critical(None, "启动错误", error_msg)
        traceback.print_exc()
        sys.exit(1)
    
    # 步骤7: 加载配置
    splash.update_progress(6, "正在加载配置...")
    # 配置已在窗口初始化时加载
    
    # 步骤8: 完成
    splash.update_progress(7, "准备就绪！")
    # 短暂延迟以便用户看到完成状态
    time.sleep(0.3)
    
    # 关闭启动画面
    splash.close()
    
    # 显示项目选择对话框（强制选择新建或加载项目）
    from src.ui.windows.startup_project_dialog import StartupProjectDialog
    startup_dialog = StartupProjectDialog(window)
    
    if not startup_dialog.exec():
        # 用户取消了对话框，退出程序
        sys.exit(0)
    
    # 处理项目选择结果
    selected_path = startup_dialog.get_selected_project_path()
    new_project_info = startup_dialog.get_new_project_info()
    
    if new_project_info:
        # 新建项目
        project_path = new_project_info['path']
        project_note = new_project_info.get('note', '')
        # 保存新项目（不显示提示框）
        success = window.save_project_with_info(project_path, project_note)
        if not success:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(window, "错误", "创建项目失败，请查看控制台输出")
            sys.exit(1)
    elif selected_path:
        # 加载项目（不显示提示框）
        success = window.project_save_manager.load_project(selected_path, window)
        if success:
            window.current_project_path = selected_path
            window.project_unsaved_changes = False
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(window, "错误", "加载项目失败，请查看控制台输出")
            sys.exit(1)
    
    # 显示主窗口
    try:
        window.show()
    except Exception as e:
        from PyQt6.QtWidgets import QMessageBox
        error_msg = f"显示主窗口失败：\n{str(e)}\n\n详细错误信息：\n{traceback.format_exc()}"
        QMessageBox.critical(None, "显示错误", error_msg)
        traceback.print_exc()
        sys.exit(1)

    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

