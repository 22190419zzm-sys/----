"""
图标管理器
用于统一管理应用程序图标
"""
import os
from pathlib import Path
from PyQt6.QtGui import QIcon


def get_resource_path(relative_path):
    """
    获取资源文件的绝对路径
    
    Args:
        relative_path: 相对于项目根目录的资源文件路径
    
    Returns:
        资源文件的绝对路径，如果文件不存在则返回None
    """
    # 获取项目根目录（main.py所在的目录）
    base_path = Path(__file__).parent.parent.parent
    resource_path = base_path / relative_path
    
    if resource_path.exists():
        return str(resource_path)
    return None


def get_app_icon():
    """
    获取应用程序图标
    
    Returns:
        QIcon对象，如果图标文件不存在则返回None
    """
    icon_path = get_resource_path("resources/icon.ico")
    if icon_path and os.path.exists(icon_path):
        return QIcon(icon_path)
    return None


def set_window_icon(window):
    """
    为窗口设置图标
    
    Args:
        window: QWidget或QDialog对象
    """
    icon = get_app_icon()
    if icon:
        window.setWindowIcon(icon)


def set_application_icon(app):
    """
    为应用程序设置图标（影响所有窗口）
    
    Args:
        app: QApplication对象
    """
    icon = get_app_icon()
    if icon:
        app.setWindowIcon(icon)

