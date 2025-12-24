import sys
import os
import warnings
import traceback

# 过滤 PyQt6 的弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*sipPyTypeDict.*')

# 首先只导入最基础的PyQt6模块，确保启动画面能立即显示
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QLinearGradient


def main():
    # 立即创建应用程序（不导入其他模块）
    app = QApplication(sys.argv)
    
    # 设置应用程序图标（影响所有窗口）
    try:
        from src.utils.icon_manager import set_application_icon
        set_application_icon(app)
    except Exception as e:
        print(f"警告: 无法加载应用程序图标: {e}")
    
    # 立即显示启动画面（使用最少的导入）
    # 直接创建启动画面窗口，不通过导入
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
    from PyQt6.QtGui import QFont, QPixmap, QPainter, QPainterPath
    from PyQt6.QtCore import QRect
    
    class SplashScreen(QWidget):
        """启动画面窗口 - 炫酷版本"""
        
        def __init__(self):
            super().__init__()
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint | 
                Qt.WindowType.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            
            # 设置窗口大小和位置（增大尺寸以容纳logo）
            self.setFixedSize(600, 400)
            self._center_window()
            
            # 尝试加载logo
            self.logo_pixmap = None
            try:
                from src.utils.icon_manager import get_resource_path
                logo_path = get_resource_path("resources/splash_logo.png")
                if logo_path and os.path.exists(logo_path):
                    self.logo_pixmap = QPixmap(logo_path)
                    # 缩放logo到合适大小（最大200x200）
                    if self.logo_pixmap.width() > 200 or self.logo_pixmap.height() > 200:
                        self.logo_pixmap = self.logo_pixmap.scaled(200, 200, 
                                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation)
            except Exception as e:
                print(f"警告: 无法加载启动画面logo: {e}")
            
            # 创建布局
            layout = QVBoxLayout(self)
            layout.setSpacing(20)
            layout.setContentsMargins(40, 40, 40, 40)
            
            # Logo标签（悬空显示，无背景）
            if self.logo_pixmap:
                self.logo_label = QLabel()
                self.logo_label.setPixmap(self.logo_pixmap)
                self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.logo_label.setStyleSheet("background: transparent; border: none;")
                layout.addWidget(self.logo_label)
            else:
                # 如果没有logo，尝试显示应用程序图标
                try:
                    from src.utils.icon_manager import get_app_icon
                    icon = get_app_icon()
                    if icon and not icon.isNull():
                        icon_label = QLabel()
                        pixmap = icon.pixmap(200, 200)
                        if not pixmap.isNull():
                            icon_label.setPixmap(pixmap)
                            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            icon_label.setStyleSheet("background: transparent; border: none;")
                            layout.addWidget(icon_label)
                except Exception as e:
                    print(f"警告: 无法加载图标: {e}")
            
            # 添加弹性空间，让logo和进度条之间有间距
            layout.addStretch()
            
            # 状态标签（隐藏，仅用于存储状态文字，不显示）
            self.status_label = QLabel("正在初始化...")
            self.status_label.hide()  # 隐藏状态标签，文字融合到进度条中
            
            # 进度条（科技风格，状态文字融合在其中）
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("正在初始化... - %p%")  # 初始格式：状态文字 + 百分比
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 rgba(0, 212, 255, 0.5), stop: 1 rgba(123, 104, 238, 0.5));
                    border-radius: 12px;
                    text-align: center;
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 rgba(10, 14, 39, 0.95), stop: 1 rgba(22, 33, 62, 0.95));
                    height: 45px;
                    font-size: 11px;
                    font-weight: bold;
                    color: #00d4ff;
                    padding: 8px;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #00d4ff, stop: 0.3 #4a90e2, stop: 0.6 #7b68ee, stop: 1 #00d4ff);
                    border-radius: 10px;
                    border: 1px solid rgba(0, 212, 255, 0.6);
                }
            """)
            layout.addWidget(self.progress_bar)
            
            # 设置窗口样式（科技感深色背景，带渐变和发光效果）
            self.setStyleSheet("""
                QWidget {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #0a0e27, 
                        stop: 0.5 #1a1f3a,
                        stop: 1 #0f1419);
                    border-radius: 15px;
                    border: 1px solid rgba(0, 212, 255, 0.3);
                }
            """)
            
            # 加载步骤列表
            self.loading_steps = []
        
        def paintEvent(self, event):
            """绘制窗口（科技风格，带发光效果）"""
            from PyQt6.QtCore import QRectF
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # 绘制外发光阴影（科技感）
            shadow_rect = QRectF(3, 3, self.width() - 6, self.height() - 6)
            shadow_path = QPainterPath()
            shadow_path.addRoundedRect(shadow_rect, 15, 15)
            # 多层阴影创造发光效果
            for i in range(3):
                alpha = 20 - i * 5
                shadow_color = QColor(0, 212, 255, alpha)
                painter.setPen(QColor(0, 0, 0, 0))
                painter.setBrush(shadow_color)
                offset = i * 2
                glow_rect = QRectF(3 - offset, 3 - offset, 
                                  self.width() - 6 + offset * 2, 
                                  self.height() - 6 + offset * 2)
                glow_path = QPainterPath()
                glow_path.addRoundedRect(glow_rect, 15, 15)
                painter.drawPath(glow_path)
            
            # 绘制主窗口（深色科技背景，带渐变）
            main_rect = QRectF(0, 0, self.width() - 3, self.height() - 3)
            path = QPainterPath()
            path.addRoundedRect(main_rect, 15, 15)
            # 使用深色渐变背景
            from PyQt6.QtGui import QLinearGradient
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(10, 14, 39))
            gradient.setColorAt(0.5, QColor(26, 26, 46))
            gradient.setColorAt(1, QColor(22, 33, 62))
            painter.fillPath(path, gradient)
            
            # 绘制顶部边框发光效果
            border_color = QColor(0, 212, 255, 150)
            painter.setPen(border_color)
            painter.setBrush(QColor(0, 0, 0, 0))
            painter.drawPath(path)
        
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
                # 保存状态文字（虽然标签已隐藏）
                self.status_label.setText(status_text)
                
                # 计算进度百分比
                if len(self.loading_steps) > 0:
                    progress = int((step_index + 1) / len(self.loading_steps) * 100)
                    self.progress_bar.setValue(progress)
                    # 进度条格式：状态文字 + 百分比（融合在一起）
                    self.progress_bar.setFormat(f"{status_text} - %p%")
            else:
                # 计算进度百分比
                if len(self.loading_steps) > 0:
                    progress = int((step_index + 1) / len(self.loading_steps) * 100)
                    self.progress_bar.setValue(progress)
                    if status_text:
                        self.progress_bar.setFormat(f"{status_text} - %p%")
            
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

