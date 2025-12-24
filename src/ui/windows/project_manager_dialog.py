"""
项目管理对话框
用于显示、管理、加载保存的项目
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QFileDialog, QLineEdit, QInputDialog, QAbstractItemView
)
from PyQt6.QtGui import QFont

from src.core.project_save_manager import ProjectSaveManager


class ProjectManagerDialog(QDialog):
    """项目管理对话框"""
    
    def __init__(self, parent=None, project_save_manager: Optional[ProjectSaveManager] = None):
        super().__init__(parent)
        self.setWindowTitle("项目管理")
        # 设置窗口图标
        try:
            from src.utils.icon_manager import set_window_icon
            set_window_icon(self)
        except:
            pass
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        self.project_save_manager = project_save_manager or ProjectSaveManager()
        self.projects_dir = self._get_projects_directory()
        self.selected_project_path = None
        
        self._setup_ui()
        self._refresh_project_list()
    
    def _get_projects_directory(self) -> Path:
        """获取项目保存目录"""
        # 在用户目录下创建项目文件夹
        projects_dir = Path.home() / "SpectraPro_Projects"
        projects_dir.mkdir(exist_ok=True)
        return projects_dir
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题和路径显示
        header_layout = QHBoxLayout()
        title_label = QLabel("项目管理")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        path_label = QLabel(f"项目目录: {self.projects_dir}")
        path_label.setStyleSheet("color: #666; font-size: 9pt;")
        header_layout.addWidget(path_label)
        layout.addLayout(header_layout)
        
        # 项目列表表格
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["项目名称", "保存日期", "文件大小", "CSV路径", "备注"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_table_double_clicked)
        layout.addWidget(self.table)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 左侧：操作按钮
        left_buttons = QHBoxLayout()
        self.btn_new = QPushButton("新建项目")
        self.btn_new.clicked.connect(self._create_new_project)
        left_buttons.addWidget(self.btn_new)
        
        self.btn_load = QPushButton("加载选中项目")
        self.btn_load.clicked.connect(self._load_selected_project)
        left_buttons.addWidget(self.btn_load)
        
        self.btn_delete = QPushButton("删除选中项目")
        self.btn_delete.clicked.connect(self._delete_selected_project)
        self.btn_delete.setStyleSheet("background-color: #f44336; color: white;")
        left_buttons.addWidget(self.btn_delete)
        
        button_layout.addLayout(left_buttons)
        button_layout.addStretch()
        
        # 右侧：关闭按钮
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_close)
        
        layout.addLayout(button_layout)
    
    def _refresh_project_list(self):
        """刷新项目列表"""
        self.table.setRowCount(0)
        
        # 扫描项目目录
        project_files = list(self.projects_dir.glob("*.json")) + list(self.projects_dir.glob("*.hdf5")) + list(self.projects_dir.glob("*.h5"))
        
        for project_file in sorted(project_files, key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                # 读取项目信息
                project_info = self._read_project_info(project_file)
                
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                # 项目名称（不含扩展名）
                name_item = QTableWidgetItem(project_file.stem)
                self.table.setItem(row, 0, name_item)
                
                # 保存日期
                save_time = project_info.get('save_time', '')
                if save_time:
                    try:
                        dt = datetime.fromisoformat(save_time)
                        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date_str = datetime.fromtimestamp(project_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = datetime.fromtimestamp(project_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                self.table.setItem(row, 1, QTableWidgetItem(date_str))
                
                # 文件大小
                size_bytes = project_file.stat().st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.2f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                self.table.setItem(row, 2, QTableWidgetItem(size_str))
                
                # CSV路径
                csv_path = project_info.get('csv_folder_path', '')
                self.table.setItem(row, 3, QTableWidgetItem(csv_path if csv_path else "(未设置)"))
                
                # 备注（从项目数据中提取，如果有的话）
                note = project_info.get('note', '')
                self.table.setItem(row, 4, QTableWidgetItem(note if note else ""))
                
                # 存储文件路径到item的data中
                name_item.setData(Qt.ItemDataRole.UserRole, str(project_file))
                
            except Exception as e:
                print(f"读取项目信息失败 {project_file}: {e}")
                continue
    
    def _read_project_info(self, project_file: Path) -> Dict[str, Any]:
        """读取项目文件的基本信息"""
        info = {}
        
        try:
            if project_file.suffix.lower() == '.json':
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info['save_time'] = data.get('save_time', '')
                    info['csv_folder_path'] = data.get('csv_folder_path', '')
                    info['note'] = data.get('note', '')
            else:
                # HDF5 格式
                try:
                    import h5py
                    with h5py.File(project_file, 'r') as f:
                        info['save_time'] = f.attrs.get('save_time', '')
                        if isinstance(info['save_time'], bytes):
                            info['save_time'] = info['save_time'].decode('utf-8')
                        info['csv_folder_path'] = f.attrs.get('csv_folder_path', '')
                        if isinstance(info['csv_folder_path'], bytes):
                            info['csv_folder_path'] = info['csv_folder_path'].decode('utf-8')
                        info['note'] = f.attrs.get('note', '')
                        if isinstance(info['note'], bytes):
                            info['note'] = info['note'].decode('utf-8')
                except ImportError:
                    pass
        except Exception as e:
            print(f"读取项目文件失败: {e}")
        
        return info
    
    def _on_table_double_clicked(self, index):
        """双击表格行时加载项目"""
        self._load_selected_project()
    
    def _load_selected_project(self):
        """加载选中的项目"""
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "警告", "请先选择一个项目")
            return
        
        name_item = self.table.item(current_row, 0)
        if not name_item:
            return
        
        project_path = name_item.data(Qt.ItemDataRole.UserRole)
        if not project_path or not os.path.exists(project_path):
            QMessageBox.critical(self, "错误", "项目文件不存在")
            return
        
        self.selected_project_path = project_path
        self.accept()
    
    def _create_new_project(self):
        """创建新项目"""
        # 获取项目名称
        name, ok = QInputDialog.getText(
            self,
            "新建项目",
            "请输入项目名称:",
            text=f"项目_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if not ok or not name.strip():
            return
        
        # 获取备注（可选）
        note, ok = QInputDialog.getText(
            self,
            "项目备注",
            "请输入项目备注（可选，可直接点击确定跳过）:",
            text=""
        )
        
        if not ok:
            return
        
        # 自动保存到项目目录，不需要手动选择文件夹
        file_path = self.projects_dir / f"{name}.json"
        
        # 如果文件已存在，询问是否覆盖
        if file_path.exists():
            reply = QMessageBox.question(
                self,
                "文件已存在",
                f"项目 '{name}' 已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # 调用父窗口的保存方法（不显示提示框）
        if self.parent() and hasattr(self.parent(), 'save_project_with_info'):
            success = self.parent().save_project_with_info(str(file_path), note if note else "")
            if not success:
                QMessageBox.critical(self, "错误", "保存项目失败，请查看控制台输出")
        elif self.parent() and hasattr(self.parent(), 'save_project'):
            # 使用默认路径保存
            success = self.parent().save_project_to_path(str(file_path), note if note else "")
            if not success:
                QMessageBox.critical(self, "错误", "保存项目失败，请查看控制台输出")
        
        # 刷新列表
        self._refresh_project_list()
    
    def _delete_selected_project(self):
        """删除选中的项目"""
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "警告", "请先选择一个项目")
            return
        
        name_item = self.table.item(current_row, 0)
        if not name_item:
            return
        
        project_path = name_item.data(Qt.ItemDataRole.UserRole)
        project_name = name_item.text()
        
        if not project_path or not os.path.exists(project_path):
            QMessageBox.critical(self, "错误", "项目文件不存在")
            return
        
        # 确认删除
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除项目 '{project_name}' 吗？\n此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(project_path)
                # 刷新列表，不显示删除成功提示
                self._refresh_project_list()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")
    
    def get_selected_project_path(self) -> Optional[str]:
        """获取选中的项目路径"""
        return self.selected_project_path

