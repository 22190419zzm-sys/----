"""
2D-COS 面板 mixin：
- 对外暴露 run_2d_cos_analysis / export_group_averages
- 内部委托给主类中的 _run_2d_cos_analysis_internal / _export_group_averages_internal
后续可将实现整体迁移到此文件。
"""
from typing import Any


import os
import glob
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QListWidget, QAbstractItemView, QHBoxLayout,
    QPushButton, QLabel, QMessageBox
)

from src.utils.helpers import group_files_by_name, natural_sort_key
from src.core.preprocessor import DataPreProcessor
from src.ui.windows.two_dcos_window import TwoDCOSWindow


class COSPanelMixin:
    def run_2d_cos_analysis(self) -> Any:
        return self._run_2d_cos_analysis_internal()

    def export_group_averages(self) -> Any:
        return self._export_group_averages_internal()

    def _run_2d_cos_analysis_internal(self):
        """
        运行2D-COS分析：基于浓度梯度数据解析重叠峰

        关键点：
        - 扰动（浓度）存在于组之间，不在组内
        - 对每个组计算平均光谱
        - 使用自然排序确保组顺序正确（如 0mg -> 25mg -> 50mg）
        """
        try:
            folder = self.folder_input.text()
            if not os.path.isdir(folder):
                QMessageBox.warning(self, "错误", "请先选择数据文件夹")
                return

            # 物理截断值（确保控件存在）
            x_min_phys = None
            x_max_phys = None
            if hasattr(self, 'x_min_phys_input') and self.x_min_phys_input:
                x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            if hasattr(self, 'x_max_phys_input') and self.x_max_phys_input:
                x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())

            # 读取基础参数
            skip = self.skip_rows_spin.value()
            n_chars = self.n_chars_spin.value()

            # 获取文件并分组
            files = sorted(glob.glob(os.path.join(folder, '*.csv')) + glob.glob(os.path.join(folder, '*.txt')))
            groups = group_files_by_name(files, n_chars)

            # 筛选指定组（如果用户指定了）
            target_gs = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}

            if len(groups) < 2:
                QMessageBox.warning(self, "错误", "2D-COS分析至少需要2个组（浓度梯度）")
                return

            # 使用自然排序对组名进行排序（关键：确保浓度顺序正确）
            initial_sorted_names = sorted(groups.keys(), key=natural_sort_key)

            # 创建手动确认组顺序的对话框
            order_dialog = QDialog(self)
            order_dialog.setWindowTitle("确认 2D-COS 浓度梯度顺序（从低到高）")
            order_dialog.setMinimumSize(400, 300)
            order_layout = QVBoxLayout(order_dialog)

            # 说明标签
            info_label = QLabel("请拖拽调整组的顺序（从上到下表示浓度从低到高）：")
            order_layout.addWidget(info_label)

            # 可拖拽排序的列表
            list_widget = QListWidget()
            list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            list_widget.addItems(initial_sorted_names)
            order_layout.addWidget(list_widget)

            # 按钮布局
            button_layout = QHBoxLayout()
            btn_ok = QPushButton("确定")
            btn_cancel = QPushButton("取消")
            btn_ok.clicked.connect(order_dialog.accept)
            btn_cancel.clicked.connect(order_dialog.reject)
            button_layout.addWidget(btn_ok)
            button_layout.addWidget(btn_cancel)
            order_layout.addLayout(button_layout)

            # 显示对话框并获取用户选择
            if order_dialog.exec() != QDialog.DialogCode.Accepted:
                return

            # 从 QListWidget 中按顺序提取最终的组名列表
            final_sorted_groups = []
            for i in range(list_widget.count()):
                final_sorted_groups.append(list_widget.item(i).text())

            # 收集每个组的平均光谱
            group_averages = []
            common_x = None

            for g_name in final_sorted_groups:
                g_files = groups[g_name]
                y_list = []
                group_x = None

                # 组内处理：收集所有有效光谱并计算平均
                for f in g_files:
                    try:
                        x, y = self.read_data(f, skip, x_min_phys, x_max_phys)
                        if group_x is None:
                            group_x = x
                        if common_x is None:
                            common_x = x

                        # 应用预处理（使用统一预处理函数）
                        preprocess_params = {
                            'qc_enabled': self.qc_check.isChecked(),
                            'qc_threshold': self.qc_threshold_spin.value(),
                            'is_be_correction': self.be_check.isChecked(),
                            'be_temp': self.be_temp_spin.value(),
                            'is_smoothing': self.smoothing_check.isChecked(),
                            'smoothing_window': self.smoothing_window_spin.value(),
                            'smoothing_poly': self.smoothing_poly_spin.value(),
                            'is_baseline_als': self.baseline_als_check.isChecked(),
                            'als_lam': self.lam_spin.value(),
                            'als_p': self.p_spin.value(),
                            'is_baseline_poly': self.baseline_poly_check.isChecked() if hasattr(self, 'baseline_poly_check') else False,
                            'baseline_points': self.baseline_points_spin.value() if hasattr(self, 'baseline_points_spin') else 50,
                            'baseline_poly': self.baseline_poly_spin.value() if hasattr(self, 'baseline_poly_spin') else 3,
                            'normalization_mode': self.normalization_combo.currentText(),
                            'global_transform_mode': self.global_transform_combo.currentText() if hasattr(self, 'global_transform_combo') else '无',
                            'global_log_base': self.global_log_base_combo.currentText() if hasattr(self, 'global_log_base_combo') else '10',
                            'global_log_offset': self.global_log_offset_spin.value() if hasattr(self, 'global_log_offset_spin') else 1.0,
                            'global_sqrt_offset': self.global_sqrt_offset_spin.value() if hasattr(self, 'global_sqrt_offset_spin') else 0.0,
                            'is_quadratic_fit': self.quadratic_fit_check.isChecked() if hasattr(self, 'quadratic_fit_check') else False,
                            'quadratic_degree': self.quadratic_degree_spin.value() if hasattr(self, 'quadratic_degree_spin') else 2,
                            'is_derivative': False,  # 2D-COS不需要二次导数
                            'global_y_offset': 0.0,  # 2D-COS不需要Y轴偏移
                        }
                        
                        # 使用统一预处理函数
                        y = DataPreProcessor.preprocess_spectrum(x, y, preprocess_params)
                        
                        # QC检查
                        if preprocess_params['qc_enabled'] and (y is None or np.max(y) < preprocess_params['qc_threshold']):
                            continue

                        # 如果X轴不一致，需要插值对齐
                        if len(x) != len(common_x) or not np.allclose(x, common_x):
                            from scipy.interpolate import interp1d
                            f_interp = interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
                            y = f_interp(common_x)

                        y_list.append(y)
                    except Exception as e:
                        print(f"警告：处理文件 {os.path.basename(f)} 时出错: {e}")
                        continue

                if not y_list:
                    print(f"警告：组 {g_name} 无有效数据，跳过")
                    continue

                # 计算该组的平均光谱
                y_array = np.array(y_list)
                y_avg = np.mean(y_array, axis=0)
                group_averages.append(y_avg)

            if len(group_averages) < 2:
                QMessageBox.warning(self, "错误", "有效组数不足（至少需要2个组）")
                return

            if common_x is None:
                QMessageBox.warning(self, "错误", "无法确定公共波数轴")
                return

            # 构建扰动矩阵 X (n_groups, n_wavenumbers)
            X_matrix = np.array(group_averages)

            # 打开2D-COS窗口
            if not hasattr(self, 'cos_window') or self.cos_window is None:
                self.cos_window = TwoDCOSWindow(self)

            self.cos_window.set_data(X_matrix, common_x, final_sorted_groups)
            self.cos_window.show()
            self.cos_window.raise_()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"2D-COS分析失败：{str(e)}")
            import traceback
            traceback.print_exc()

    def _export_group_averages_internal(self):
        """导出组瀑布图中所有组的平均值谱线"""
        try:
            folder_path = self.folder_input.text()
            if not os.path.isdir(folder_path):
                QMessageBox.warning(self, "警告", "请先设置数据文件夹路径")
                return

            # 获取分组参数
            skip = self.skip_rows_spin.value()
            n_chars = self.n_chars_spin.value()
            x_min_phys = self._parse_optional_float(self.x_min_phys_input.text())
            x_max_phys = self._parse_optional_float(self.x_max_phys_input.text())

            # 扫描文件并分组
            files = sorted(glob.glob(os.path.join(folder_path, '*.csv')) + glob.glob(os.path.join(folder_path, '*.txt')))
            if not files:
                QMessageBox.warning(self, "警告", "未找到文件")
                return

            groups = group_files_by_name(files, n_chars)

            # 筛选指定组
            target_gs = [x.strip() for x in self.groups_input.text().split(',') if x.strip()]
            if target_gs:
                groups = {k: v for k, v in groups.items() if k in target_gs}

            if not groups:
                QMessageBox.warning(self, "警告", "未找到有效的组")
                return

            # 物理截断
            common_x = None
            export_data = {}
            for g_name, g_files in groups.items():
                y_list = []
                for f in g_files:
                    try:
                        x, y = self.read_data(f, skip, x_min_phys, x_max_phys)  # 使用物理截断
                        if common_x is None:
                            common_x = x
                        # 对齐到 common_x
                        if len(x) != len(common_x) or not np.allclose(x, common_x):
                            from scipy.interpolate import interp1d
                            f_interp = interp1d(x, y, kind='linear', fill_value=0, bounds_error=False)
                            y = f_interp(common_x)
                        y_list.append(y)
                    except Exception as e:
                        print(f"警告：处理文件 {os.path.basename(f)} 时出错: {e}")
                        continue

                if not y_list:
                    print(f"警告：组 {g_name} 无有效数据，跳过")
                    continue

                y_array = np.array(y_list)
                y_avg = np.mean(y_array, axis=0)
                export_data[g_name] = y_avg

            if not export_data or common_x is None:
                QMessageBox.warning(self, "警告", "无有效数据可导出")
                return

            # 选择保存目录
            save_dir = self._choose_save_directory()
            if not save_dir:
                return

            import pandas as pd
            count = 0
            for g_name, y_avg in export_data.items():
                df = pd.DataFrame({'Wavenumber': common_x, 'Intensity': y_avg})
                out_name = f"avg_{g_name}.csv"
                df.to_csv(os.path.join(save_dir, out_name), index=False)
                count += 1

            QMessageBox.information(self, "完成", f"已导出 {count} 个组的平均光谱。")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

