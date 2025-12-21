"""
项目存档管理器
支持保存和加载完整的项目状态，包括：
- CSV 路径
- PlotConfig（所有绘图参数）
- 峰值匹配结果
- 各层数据状态（预处理后的数据、NMF结果等）
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from src.core.plot_config_manager import PlotConfig, PlotConfigManager


class ProjectSaveManager:
    """项目存档管理器"""
    
    def __init__(self):
        self.version = "1.0"
        self.projects_dir = self._get_projects_directory()
    
    def _get_projects_directory(self) -> Path:
        """获取项目保存目录"""
        projects_dir = Path.home() / "SpectraPro_Projects"
        projects_dir.mkdir(exist_ok=True)
        return projects_dir
    
    def save_project(self, file_path: str, main_window, note: Optional[str] = None) -> bool:
        """
        保存项目到文件
        
        Args:
            file_path: 保存路径（.json 或 .hdf5）
            main_window: 主窗口对象
            
        Returns:
            是否保存成功
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.json':
                return self._save_as_json(file_path, main_window, note)
            elif file_ext in ['.hdf5', '.h5']:
                return self._save_as_hdf5(file_path, main_window, note)
            else:
                # 默认使用 JSON
                return self._save_as_json(file_path, main_window, note)
        except Exception as e:
            print(f"保存项目失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_project(self, file_path: str, main_window) -> bool:
        """
        从文件加载项目
        
        Args:
            file_path: 文件路径
            main_window: 主窗口对象
            
        Returns:
            是否加载成功
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.json':
                return self._load_from_json(file_path, main_window)
            elif file_ext in ['.hdf5', '.h5']:
                return self._load_from_hdf5(file_path, main_window)
            else:
                # 默认使用 JSON
                return self._load_from_json(file_path, main_window)
        except Exception as e:
            print(f"加载项目失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_as_json(self, file_path: str, main_window, note: Optional[str] = None) -> bool:
        """保存为 JSON 格式"""
        project_data = self._collect_project_data(main_window, note)
        
        # 转换 numpy 数组为列表
        project_data = self._convert_numpy_to_list(project_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    def _load_from_json(self, file_path: str, main_window) -> bool:
        """从 JSON 格式加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
        
        # 恢复 numpy 数组
        project_data = self._convert_list_to_numpy(project_data)
        
        return self._restore_project_data(project_data, main_window)
    
    def _save_as_hdf5(self, file_path: str, main_window, note: Optional[str] = None) -> bool:
        """保存为 HDF5 格式"""
        try:
            import h5py
        except ImportError:
            print("警告: h5py 未安装，无法使用 HDF5 格式，改用 JSON")
            return self._save_as_json(file_path.replace('.hdf5', '.json').replace('.h5', '.json'), main_window, note)
        
        project_data = self._collect_project_data(main_window, note)
        
        with h5py.File(file_path, 'w') as f:
            # 保存元数据
            f.attrs['version'] = self.version
            f.attrs['save_time'] = datetime.now().isoformat()
            f.attrs['note'] = note or ''
            
            # 保存配置（JSON 格式）
            config_dict = project_data.get('plot_config', {})
            f.attrs['plot_config'] = json.dumps(config_dict, ensure_ascii=False)
            
            # 保存路径信息
            f.attrs['csv_folder_path'] = project_data.get('csv_folder_path', '')
            
            # 保存峰值匹配结果
            if 'peak_matching_results' in project_data:
                peak_group = f.create_group('peak_matching_results')
                self._save_dict_to_hdf5(peak_group, project_data['peak_matching_results'])
            
            # 保存数据状态
            if 'data_states' in project_data:
                data_group = f.create_group('data_states')
                self._save_dict_to_hdf5(data_group, project_data['data_states'])
            
            # 保存其他信息
            if 'other_info' in project_data:
                other_group = f.create_group('other_info')
                self._save_dict_to_hdf5(other_group, project_data['other_info'])
        
        return True
    
    def _load_from_hdf5(self, file_path: str, main_window) -> bool:
        """从 HDF5 格式加载"""
        try:
            import h5py
        except ImportError:
            print("警告: h5py 未安装，无法使用 HDF5 格式")
            return False
        
        with h5py.File(file_path, 'r') as f:
            project_data = {}
            
            # 加载元数据
            project_data['version'] = f.attrs.get('version', '1.0')
            project_data['save_time'] = f.attrs.get('save_time', '')
            
            # 加载配置
            if 'plot_config' in f.attrs:
                config_json = f.attrs['plot_config']
                project_data['plot_config'] = json.loads(config_json)
            
            # 加载路径信息
            project_data['csv_folder_path'] = f.attrs.get('csv_folder_path', '')
            
            # 加载峰值匹配结果
            if 'peak_matching_results' in f:
                project_data['peak_matching_results'] = self._load_dict_from_hdf5(f['peak_matching_results'])
            
            # 加载数据状态
            if 'data_states' in f:
                project_data['data_states'] = self._load_dict_from_hdf5(f['data_states'])
            
            # 加载其他信息
            if 'other_info' in f:
                project_data['other_info'] = self._load_dict_from_hdf5(f['other_info'])
        
        return self._restore_project_data(project_data, main_window)
    
    def _collect_project_data(self, main_window, note: Optional[str] = None) -> Dict[str, Any]:
        """收集项目数据"""
        project_data = {
            'version': self.version,
            'save_time': datetime.now().isoformat(),
            'note': note or '',
        }
        
        # 1. CSV 路径
        if hasattr(main_window, 'folder_input'):
            project_data['csv_folder_path'] = main_window.folder_input.text()
        else:
            project_data['csv_folder_path'] = ''
        
        # 2. PlotConfig（所有绘图参数）
        try:
            config_manager = PlotConfigManager()
            config = config_manager.get_config()
            project_data['plot_config'] = config.to_dict()
        except Exception as e:
            print(f"收集 PlotConfig 失败: {e}")
            project_data['plot_config'] = {}
        
        # 3. 峰值匹配结果
        project_data['peak_matching_results'] = self._collect_peak_matching_results(main_window)
        
        # 4. 各层数据状态
        project_data['data_states'] = self._collect_data_states(main_window)
        
        # 5. 其他信息（窗口状态、控件值等）
        project_data['other_info'] = self._collect_other_info(main_window)
        
        return project_data
    
    def _collect_peak_matching_results(self, main_window) -> Dict[str, Any]:
        """收集峰值匹配结果"""
        results = {}
        
        try:
            # 从峰值匹配面板获取结果
            if hasattr(main_window, 'peak_matching_panel') and main_window.peak_matching_panel:
                panel = main_window.peak_matching_panel
                # 如果有匹配结果，保存它们
                if hasattr(panel, 'last_matching_results'):
                    results['last_matching_results'] = panel.last_matching_results
        except Exception as e:
            print(f"收集峰值匹配结果失败: {e}")
        
        return results
    
    def _collect_data_states(self, main_window) -> Dict[str, Any]:
        """收集各层数据状态"""
        states = {}
        
        try:
            # NMF 结果
            if hasattr(main_window, 'nmf_window') and main_window.nmf_window:
                nmf_win = main_window.nmf_window
                if hasattr(nmf_win, 'W') and hasattr(nmf_win, 'H'):
                    states['nmf'] = {
                        'W': nmf_win.W.tolist() if isinstance(nmf_win.W, np.ndarray) else nmf_win.W,
                        'H': nmf_win.H.tolist() if isinstance(nmf_win.H, np.ndarray) else nmf_win.H,
                        'common_x': nmf_win.common_x.tolist() if hasattr(nmf_win, 'common_x') and isinstance(nmf_win.common_x, np.ndarray) else [],
                        'sample_labels': nmf_win.sample_labels if hasattr(nmf_win, 'sample_labels') else [],
                    }
            
            # 预处理后的数据
            if hasattr(main_window, 'processed_data'):
                # 只保存元数据，不保存完整数据（避免文件过大）
                states['processed_data_metadata'] = {
                    'count': len(main_window.processed_data) if main_window.processed_data else 0,
                }
            
            # 绘图窗口数据和状态
            if hasattr(main_window, 'plot_windows'):
                plot_windows_data = {}
                for name, window in main_window.plot_windows.items():
                    if window:
                        window_data = {
                            'geometry': {
                                'x': window.x(),
                                'y': window.y(),
                                'width': window.width(),
                                'height': window.height(),
                            },
                            'is_visible': window.isVisible(),
                            'group_name': window.group_name if hasattr(window, 'group_name') else name,
                        }
                        
                        # 保存绘图数据
                        if hasattr(window, 'current_plot_data'):
                            plot_data_dict = {}
                            for key, data in window.current_plot_data.items():
                                if isinstance(data, dict):
                                    plot_data_dict[key] = {
                                        'x': data.get('x', []).tolist() if isinstance(data.get('x'), np.ndarray) else data.get('x', []),
                                        'y': data.get('y', []).tolist() if isinstance(data.get('y'), np.ndarray) else data.get('y', []),
                                        'label': data.get('label', ''),
                                        'color': data.get('color', 'gray'),
                                        'type': data.get('type', 'Individual'),
                                        'linewidth': data.get('linewidth', 1.2),
                                        'linestyle': data.get('linestyle', '-'),
                                        'shadow_upper': data.get('shadow_upper', []).tolist() if isinstance(data.get('shadow_upper'), np.ndarray) else data.get('shadow_upper', []),
                                        'shadow_lower': data.get('shadow_lower', []).tolist() if isinstance(data.get('shadow_lower'), np.ndarray) else data.get('shadow_lower', []),
                                    }
                            window_data['plot_data'] = plot_data_dict
                        
                        # 保存最后使用的plot_params（用于恢复时重新绘制）
                        if hasattr(window, '_last_plot_params'):
                            # 转换numpy数组为列表
                            last_params = window._last_plot_params.copy()
                            # 移除大的数据数组，只保留元数据
                            if 'grouped_files_data' in last_params:
                                # 只保存文件名和元数据，不保存完整数据
                                grouped_files_meta = []
                                for item in last_params['grouped_files_data']:
                                    if isinstance(item, tuple) and len(item) >= 2:
                                        file_path, x_data, y_data = item[0], item[1], item[2] if len(item) > 2 else None
                                        grouped_files_meta.append({
                                            'file_path': str(file_path),
                                            'x_len': len(x_data) if isinstance(x_data, np.ndarray) else 0,
                                            'y_len': len(y_data) if isinstance(y_data, np.ndarray) else 0,
                                        })
                                last_params['grouped_files_data'] = grouped_files_meta
                            window_data['last_plot_params'] = self._convert_numpy_to_list(last_params)
                        
                        plot_windows_data[name] = window_data
                states['plot_windows'] = plot_windows_data
            
            # 其他窗口状态（NMF窗口、定量窗口等）
            if hasattr(main_window, 'nmf_window') and main_window.nmf_window:
                nmf_win = main_window.nmf_window
                states['nmf_window_state'] = {
                    'geometry': {
                        'x': nmf_win.x(),
                        'y': nmf_win.y(),
                        'width': nmf_win.width(),
                        'height': nmf_win.height(),
                    },
                    'is_visible': nmf_win.isVisible(),
                }
            
            if hasattr(main_window, 'quantitative_window') and main_window.quantitative_window:
                qty_win = main_window.quantitative_window
                states['quantitative_window_state'] = {
                    'geometry': {
                        'x': qty_win.x(),
                        'y': qty_win.y(),
                        'width': qty_win.width(),
                        'height': qty_win.height(),
                    },
                    'is_visible': qty_win.isVisible(),
                }
        except Exception as e:
            print(f"收集数据状态失败: {e}")
            import traceback
            traceback.print_exc()
        
        return states
    
    def _safe_get_widget_text(self, widget):
        """安全获取 widget 的文本，如果 widget 已被删除则返回空字符串"""
        try:
            if widget is None:
                return ""
            # 检查 widget 是否仍然有效
            if not hasattr(widget, 'text'):
                return ""
            return widget.text().strip()
        except RuntimeError:
            # widget 已被删除
            return ""
        except Exception:
            return ""
    
    def _collect_other_info(self, main_window) -> Dict[str, Any]:
        """收集其他信息"""
        info = {}
        
        try:
            # 窗口几何信息
            info['window_geometry'] = {
                'x': main_window.x(),
                'y': main_window.y(),
                'width': main_window.width(),
                'height': main_window.height(),
            }
            
            # 图例重命名
            if hasattr(main_window, 'legend_rename_widgets'):
                legend_renames = {}
                try:
                    for key, widget in list(main_window.legend_rename_widgets.items()):
                        text = self._safe_get_widget_text(widget)
                        if text:
                            legend_renames[key] = text
                except (RuntimeError, AttributeError):
                    pass
                info['legend_renames'] = legend_renames
            
            # NMF 组分重命名
            if hasattr(main_window, 'nmf_component_rename_widgets'):
                nmf_renames = {}
                try:
                    for key, widget in list(main_window.nmf_component_rename_widgets.items()):
                        text = self._safe_get_widget_text(widget)
                        if text:
                            nmf_renames[key] = text
                except (RuntimeError, AttributeError):
                    pass
                info['nmf_component_renames'] = nmf_renames
            
            # 组瀑布图控制
            if hasattr(main_window, 'group_waterfall_control_widgets'):
                waterfall_controls = {}
                try:
                    for group_name, widgets in list(main_window.group_waterfall_control_widgets.items()):
                        try:
                            offset = 0.0
                            color = ''
                            if widgets.get('offset'):
                                try:
                                    offset = widgets['offset'].value()
                                except (RuntimeError, AttributeError):
                                    pass
                            if widgets.get('color'):
                                color = self._safe_get_widget_text(widgets['color'])
                            waterfall_controls[group_name] = {
                                'offset': offset,
                                'color': color,
                            }
                        except (RuntimeError, AttributeError):
                            continue
                except (RuntimeError, AttributeError):
                    pass
                info['waterfall_controls'] = waterfall_controls
        except Exception as e:
            print(f"收集其他信息失败: {e}")
            import traceback
            traceback.print_exc()
        
        return info
    
    def _restore_project_data(self, project_data: Dict[str, Any], main_window) -> bool:
        """恢复项目数据"""
        print("[DEBUG] 开始恢复项目数据...")
        try:
            # 1. 恢复 CSV 路径（但不触发文件夹改变事件，避免UI问题）
            if 'csv_folder_path' in project_data:
                csv_path = project_data['csv_folder_path']
                print(f"[DEBUG] 恢复CSV路径: {csv_path}")
                if hasattr(main_window, 'folder_input'):
                    main_window.folder_input.setText(csv_path)
                    print(f"[DEBUG] CSV路径已设置到folder_input")
                    # 注意：不触发_on_folder_changed()，避免在加载时触发UI更新导致问题
                    # 用户需要手动重新运行绘图
            
            # 2. 恢复 PlotConfig
            if 'plot_config' in project_data:
                print("[DEBUG] 恢复PlotConfig...")
                try:
                    config_manager = PlotConfigManager()
                    config = PlotConfig.from_dict(project_data['plot_config'])
                    config_manager.update_config(config)
                    print("[DEBUG] PlotConfig已更新到配置管理器")
                    
                    # 更新 UI（延迟到UI完全初始化后）
                    # 不在这里立即更新，避免UI未完全初始化时出错
                    print("[DEBUG] PlotConfig恢复完成，UI将在窗口显示后更新")
                except Exception as e:
                    print(f"[ERROR] 恢复 PlotConfig 失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 3. 恢复峰值匹配结果
            if 'peak_matching_results' in project_data:
                print("[DEBUG] 恢复峰值匹配结果...")
                self._restore_peak_matching_results(project_data['peak_matching_results'], main_window)
            
            # 4. 恢复数据状态（延迟到窗口显示后）
            if 'data_states' in project_data:
                print("[DEBUG] 准备恢复数据状态（将在窗口显示后执行）...")
                # 保存project_data以便后续使用
                main_window._pending_project_data_states = project_data['data_states']
                main_window._pending_project_data = project_data
            
            # 5. 恢复其他信息
            if 'other_info' in project_data:
                print("[DEBUG] 恢复其他信息...")
                self._restore_other_info(project_data['other_info'], main_window)
            
            print("[DEBUG] 项目数据恢复完成")
            return True
        except Exception as e:
            print(f"[ERROR] 恢复项目数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _restore_peak_matching_results(self, results: Dict[str, Any], main_window):
        """恢复峰值匹配结果"""
        try:
            if hasattr(main_window, 'peak_matching_panel') and main_window.peak_matching_panel:
                panel = main_window.peak_matching_panel
                if 'last_matching_results' in results:
                    panel.last_matching_results = results['last_matching_results']
        except Exception as e:
            print(f"恢复峰值匹配结果失败: {e}")
    
    def _restore_data_states(self, states: Dict[str, Any], main_window, project_data: Dict[str, Any] = None):
        """恢复数据状态"""
        print("[DEBUG] _restore_data_states: 开始恢复数据状态...")
        try:
            if project_data is None:
                project_data = {}
            print(f"[DEBUG] _restore_data_states: 数据状态键: {list(states.keys())}")
            
            # NMF 结果
            if 'nmf' in states:
                nmf_data = states['nmf']
                # 保存NMF数据，供后续恢复窗口使用
                if not hasattr(main_window, '_saved_nmf_data'):
                    main_window._saved_nmf_data = {}
                main_window._saved_nmf_data = nmf_data
                
                # 如果NMF窗口存在，尝试恢复NMF窗口状态
                if hasattr(main_window, 'nmf_window') and main_window.nmf_window:
                    nmf_win = main_window.nmf_window
                    if 'W' in nmf_data and 'H' in nmf_data:
                        import numpy as np
                        W = np.array(nmf_data['W'])
                        H = np.array(nmf_data['H'])
                        common_x = np.array(nmf_data.get('common_x', []))
                        sample_labels = nmf_data.get('sample_labels', [])
                        
                        # 恢复NMF窗口数据
                        nmf_win.W = W
                        nmf_win.H = H
                        if len(common_x) > 0:
                            nmf_win.common_x = common_x
                        if sample_labels:
                            nmf_win.sample_labels = sample_labels
                        
                        # 重新绘制NMF结果
                        if hasattr(nmf_win, 'plot_results'):
                            # 获取样式参数
                            from src.core.plot_config_manager import PlotConfigManager
                            config_manager = PlotConfigManager()
                            config = config_manager.get_config()
                            style_params = config.to_dict()
                            nmf_win.plot_results(style_params)
            
            # 恢复绘图窗口状态和绘图数据
            if 'plot_windows' in states:
                print("[DEBUG] _restore_data_states: 开始恢复绘图窗口...")
                plot_windows_data = states['plot_windows']
                print(f"[DEBUG] _restore_data_states: 需要恢复的窗口数量: {len(plot_windows_data)}")
                
                if hasattr(main_window, 'plot_windows'):
                    from src.ui.windows.plot_window import MplPlotWindow
                    import numpy as np
                    
                    for name, window_data in plot_windows_data.items():
                        print(f"[DEBUG] _restore_data_states: 恢复窗口 '{name}'...")
                        # 如果窗口不存在，先创建窗口
                        if name not in main_window.plot_windows:
                            group_name = window_data.get('group_name', name)
                            # 创建窗口，使用保存的几何信息
                            if 'geometry' in window_data:
                                geom = window_data['geometry']
                                initial_geometry = (
                                    geom.get('x', 100),
                                    geom.get('y', 100),
                                    geom.get('width', 800),
                                    geom.get('height', 600)
                                )
                            else:
                                initial_geometry = None
                            main_window.plot_windows[name] = MplPlotWindow(group_name, initial_geometry=initial_geometry, parent=main_window)
                        
                        window = main_window.plot_windows[name]
                        if window:
                            # 恢复窗口几何信息
                            if 'geometry' in window_data:
                                geom = window_data['geometry']
                                window.setGeometry(
                                    geom.get('x', 100),
                                    geom.get('y', 100),
                                    geom.get('width', 800),
                                    geom.get('height', 600)
                                )
                            
                            # 恢复绘图数据
                            if 'plot_data' in window_data and hasattr(window, 'current_plot_data'):
                                for key, data in window_data['plot_data'].items():
                                    if isinstance(data, dict):
                                        window.current_plot_data[key] = {
                                            'x': np.array(data.get('x', [])),
                                            'y': np.array(data.get('y', [])),
                                            'label': data.get('label', ''),
                                            'color': data.get('color', 'gray'),
                                            'type': data.get('type', 'Individual'),
                                            'linewidth': data.get('linewidth', 1.2),
                                            'linestyle': data.get('linestyle', '-'),
                                            'shadow_upper': np.array(data.get('shadow_upper', [])) if data.get('shadow_upper') else None,
                                            'shadow_lower': np.array(data.get('shadow_lower', [])) if data.get('shadow_lower') else None,
                                        }
                            
                            # 如果有保存的plot_params，尝试重新绘制
                            if 'last_plot_params' in window_data:
                                try:
                                    # 恢复plot_params
                                    saved_params = window_data['last_plot_params']
                                    # 从保存的元数据重建grouped_files_data
                                    if 'grouped_files_data' in saved_params:
                                        grouped_files_meta = saved_params['grouped_files_data']
                                        # 尝试从CSV文件夹重新读取数据
                                        csv_folder = project_data.get('csv_folder_path', '')
                                        if csv_folder and os.path.isdir(csv_folder):
                                            # 从current_plot_data重建grouped_files_data
                                            grouped_files_data = []
                                            for key, plot_data in window.current_plot_data.items():
                                                if isinstance(plot_data, dict) and 'x' in plot_data and 'y' in plot_data:
                                                    # 创建一个元组 (file_path, x, y)
                                                    # file_path可以是标签或key
                                                    file_path = plot_data.get('label', key)
                                                    x_data = plot_data['x']
                                                    y_data = plot_data['y']
                                                    grouped_files_data.append((file_path, x_data, y_data))
                                            
                                            if grouped_files_data:
                                                saved_params['grouped_files_data'] = grouped_files_data
                                    
                                    # 恢复plot_params到窗口
                                    window._last_plot_params = saved_params
                                    
                                    # 尝试重新绘制
                                    if hasattr(window, 'update_plot'):
                                        window.update_plot(saved_params)
                                        print(f"已恢复并重新绘制窗口 '{name}'")
                                except Exception as e:
                                    print(f"恢复窗口 '{name}' 的绘图失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                            elif window.current_plot_data:
                                # 如果没有保存的plot_params，但有绘图数据，尝试从current_plot_data重新绘制
                                try:
                                    # 从current_plot_data重建简单的plot_params
                                    simple_params = main_window._prepare_plot_params(grouped_files_data=None, control_data_list=None)
                                    if simple_params:
                                        # 从current_plot_data重建grouped_files_data
                                        grouped_files_data = []
                                        for key, plot_data in window.current_plot_data.items():
                                            if isinstance(plot_data, dict) and 'x' in plot_data and 'y' in plot_data:
                                                file_path = plot_data.get('label', key)
                                                x_data = plot_data['x']
                                                y_data = plot_data['y']
                                                grouped_files_data.append((file_path, x_data, y_data))
                                        
                                        if grouped_files_data:
                                            simple_params['grouped_files_data'] = grouped_files_data
                                            window.update_plot(simple_params)
                                            print(f"已从绘图数据恢复窗口 '{name}'")
                                except Exception as e:
                                    print(f"从绘图数据恢复窗口 '{name}' 失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            # 恢复窗口可见性
                            if window_data.get('is_visible', False):
                                window.show()
            
            # 恢复其他窗口状态
            if 'nmf_window_state' in states:
                nmf_state = states['nmf_window_state']
                if hasattr(main_window, 'nmf_window') and main_window.nmf_window:
                    nmf_win = main_window.nmf_window
                    if 'geometry' in nmf_state:
                        geom = nmf_state['geometry']
                        nmf_win.setGeometry(
                            geom.get('x', 100),
                            geom.get('y', 100),
                            geom.get('width', 800),
                            geom.get('height', 600)
                        )
                    if nmf_state.get('is_visible', False):
                        nmf_win.show()
            
            if 'quantitative_window_state' in states:
                qty_state = states['quantitative_window_state']
                if hasattr(main_window, 'quantitative_window') and main_window.quantitative_window:
                    qty_win = main_window.quantitative_window
                    if 'geometry' in qty_state:
                        geom = qty_state['geometry']
                        qty_win.setGeometry(
                            geom.get('x', 100),
                            geom.get('y', 100),
                            geom.get('width', 800),
                            geom.get('height', 600)
                        )
                    if qty_state.get('is_visible', False):
                        qty_win.show()
        except Exception as e:
            print(f"恢复数据状态失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _restore_other_info(self, info: Dict[str, Any], main_window):
        """恢复其他信息"""
        try:
            # 窗口几何信息
            if 'window_geometry' in info:
                geom = info['window_geometry']
                main_window.setGeometry(geom['x'], geom['y'], geom['width'], geom['height'])
            
            # 图例重命名
            if 'legend_renames' in info:
                if hasattr(main_window, 'legend_rename_widgets'):
                    for key, value in info['legend_renames'].items():
                        try:
                            if key in main_window.legend_rename_widgets:
                                widget = main_window.legend_rename_widgets[key]
                                if widget and hasattr(widget, 'setText'):
                                    widget.setText(value)
                        except (RuntimeError, AttributeError):
                            continue
            
            # NMF 组分重命名
            if 'nmf_component_renames' in info:
                if hasattr(main_window, 'nmf_component_rename_widgets'):
                    for key, value in info['nmf_component_renames'].items():
                        try:
                            if key in main_window.nmf_component_rename_widgets:
                                widget = main_window.nmf_component_rename_widgets[key]
                                if widget and hasattr(widget, 'setText'):
                                    widget.setText(value)
                        except (RuntimeError, AttributeError):
                            continue
            
            # 组瀑布图控制
            if 'waterfall_controls' in info:
                if hasattr(main_window, 'group_waterfall_control_widgets'):
                    for group_name, controls in info['waterfall_controls'].items():
                        try:
                            if group_name in main_window.group_waterfall_control_widgets:
                                widgets = main_window.group_waterfall_control_widgets[group_name]
                                if 'offset' in controls and widgets.get('offset'):
                                    try:
                                        widgets['offset'].setValue(controls['offset'])
                                    except (RuntimeError, AttributeError):
                                        pass
                                if 'color' in controls and widgets.get('color'):
                                    try:
                                        widgets['color'].setText(controls['color'])
                                    except (RuntimeError, AttributeError):
                                        pass
                        except (RuntimeError, AttributeError):
                            continue
        except Exception as e:
            print(f"恢复其他信息失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _convert_numpy_to_list(self, obj):
        """递归转换 numpy 数组为列表"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def _convert_list_to_numpy(self, obj):
        """递归转换列表为 numpy 数组（仅在需要时）"""
        if isinstance(obj, dict):
            return {k: self._convert_list_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # 检查是否是数字列表（可能是 numpy 数组）
            if len(obj) > 0 and isinstance(obj[0], (int, float)):
                try:
                    return np.array(obj)
                except:
                    return [self._convert_list_to_numpy(item) for item in obj]
            else:
                return [self._convert_list_to_numpy(item) for item in obj]
        else:
            return obj
    
    def _save_dict_to_hdf5(self, group, data: Dict[str, Any]):
        """将字典保存到 HDF5 组"""
        import h5py
        
        for key, value in data.items():
            if isinstance(value, dict):
                sub_group = group.create_group(key)
                self._save_dict_to_hdf5(sub_group, value)
            elif isinstance(value, (list, np.ndarray)):
                if len(value) > 0 and isinstance(value[0], (int, float)):
                    group.create_dataset(key, data=np.array(value))
                else:
                    # 复杂列表，保存为 JSON
                    group.attrs[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)):
                group.attrs[key] = value
            else:
                # 其他类型，保存为 JSON
                group.attrs[key] = json.dumps(value, ensure_ascii=False)
    
    def _load_dict_from_hdf5(self, group) -> Dict[str, Any]:
        """从 HDF5 组加载字典"""
        import h5py
        
        result = {}
        
        # 加载属性
        for key in group.attrs:
            value = group.attrs[key]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            # 尝试解析 JSON
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    value = json.loads(value)
                except:
                    pass
            result[key] = value
        
        # 加载数据集和子组
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                result[key] = self._load_dict_from_hdf5(item)
            elif isinstance(item, h5py.Dataset):
                result[key] = item[:].tolist()
        
        return result

