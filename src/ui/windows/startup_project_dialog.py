"""
启动时的项目选择对话框
强制用户选择新建项目或加载项目
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QMessageBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from pathlib import Path
from datetime import datetime
import json

from src.core.project_save_manager import ProjectSaveManager


class StartupProjectDialog(QDialog):
    """启动时的项目选择对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择项目")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        self.project_save_manager = ProjectSaveManager()
        self.projects_dir = self.project_save_manager._get_projects_directory()
        self.selected_project_path = None
        self.new_project_name = None
        
        self._setup_ui()
        self._load_recent_projects()
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title_label = QLabel("欢迎使用光谱数据处理工作站")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 说明文字
        desc_label = QLabel("请选择新建项目或加载已有项目")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("color: #666; font-size: 11pt;")
        layout.addWidget(desc_label)
        
        layout.addSpacing(10)
        
        # 最近项目列表
        list_label = QLabel("最近项目:")
        list_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        layout.addWidget(list_label)
        
        self.project_list = QListWidget()
        self.project_list.setMaximumHeight(250)
        self.project_list.itemDoubleClicked.connect(self._on_project_double_clicked)
        layout.addWidget(self.project_list)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 新建项目按钮
        self.btn_new = QPushButton("新建项目")
        self.btn_new.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.btn_new.clicked.connect(self._create_new_project)
        button_layout.addWidget(self.btn_new)
        
        button_layout.addStretch()
        
        # 加载项目按钮
        self.btn_load = QPushButton("加载选中项目")
        self.btn_load.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                padding: 10px 20px;
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.btn_load.clicked.connect(self._load_selected_project)
        button_layout.addWidget(self.btn_load)
        
        button_layout.addStretch()
        
        # 删除项目按钮
        self.btn_delete = QPushButton("删除选中项目")
        self.btn_delete.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                padding: 10px 20px;
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.btn_delete.clicked.connect(self._delete_selected_project)
        button_layout.addWidget(self.btn_delete)
        
        layout.addLayout(button_layout)
    
    def _load_recent_projects(self):
        """加载最近的项目列表"""
        self.project_list.clear()
        
        # 扫描项目目录
        project_files = list(self.projects_dir.glob("*.json")) + list(self.projects_dir.glob("*.hdf5")) + list(self.projects_dir.glob("*.h5"))
        
        # 按修改时间排序
        project_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # 只显示最近10个项目
        for project_file in project_files[:10]:
            try:
                # 读取项目信息
                project_info = self._read_project_info(project_file)
                
                name = project_file.stem
                save_time = project_info.get('save_time', '')
                if save_time:
                    try:
                        dt = datetime.fromisoformat(save_time)
                        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date_str = datetime.fromtimestamp(project_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = datetime.fromtimestamp(project_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                csv_path = project_info.get('csv_folder_path', '')
                note = project_info.get('note', '')
                
                # 显示格式：名称 | 日期 | CSV路径 | 备注
                display_text = f"{name} | {date_str}"
                if csv_path:
                    display_text += f" | {csv_path}"
                if note:
                    display_text += f" | {note}"
                
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, str(project_file))
                self.project_list.addItem(item)
            except Exception as e:
                print(f"读取项目信息失败 {project_file}: {e}")
                continue
    
    def _read_project_info(self, project_file: Path) -> dict:
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
    
    def _on_project_double_clicked(self, item):
        """双击项目时加载"""
        self._load_selected_project()
    
    def _load_selected_project(self):
        """加载选中的项目"""
        current_item = self.project_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个项目")
            return
        
        project_path = current_item.data(Qt.ItemDataRole.UserRole)
        if not project_path or not Path(project_path).exists():
            QMessageBox.critical(self, "错误", "项目文件不存在")
            return
        
        self.selected_project_path = project_path
        self.accept()
    
    def _create_new_project(self):
        """创建新项目"""
        from PyQt6.QtWidgets import QInputDialog
        
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
        
        # 自动保存到项目目录
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
        
        self.new_project_name = name
        self.new_project_path = str(file_path)
        self.new_project_note = note if note else ""
        self.accept()
    
    def get_selected_project_path(self):
        """获取选中的项目路径"""
        return self.selected_project_path
    
    def get_new_project_info(self):
        """获取新建项目的信息"""
        if hasattr(self, 'new_project_path'):
            return {
                'path': self.new_project_path,
                'name': self.new_project_name,
                'note': self.new_project_note
            }
        return None
    
    def _delete_selected_project(self):
        """删除选中的项目"""
        import os
        
        current_item = self.project_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "警告", "请先选择一个项目")
            return
        
        project_path = current_item.data(Qt.ItemDataRole.UserRole)
        if not project_path or not Path(project_path).exists():
            QMessageBox.critical(self, "错误", "项目文件不存在")
            return
        
        # 确认删除
        project_name = Path(project_path).stem
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
                # 刷新列表
                self._load_recent_projects()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {e}")

