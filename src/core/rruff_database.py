"""
RRUFF数据库管理系统
支持将预处理后的RRUFF库保存为数据库，支持手动选择和自动识别
"""
import os
import pickle
import hashlib
import json
import sqlite3
from typing import Dict, Optional, List
import numpy as np


class RRUFFDatabase:
    """RRUFF数据库管理器"""
    
    def __init__(self, db_dir=None):
        """
        Args:
            db_dir: 数据库目录（默认在用户目录下的.spectrapro_db）
        """
        if db_dir is None:
            db_dir = os.path.join(os.path.expanduser("~"), ".spectrapro_db")
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        
        # 数据库索引文件（SQLite）
        self.index_db_path = os.path.join(self.db_dir, "rruff_index.db")
        self._init_index_db()
    
    def _init_index_db(self):
        """初始化数据库索引"""
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS databases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                folder_path TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                preprocess_params TEXT NOT NULL,
                peak_detection_params TEXT,
                spectra_count INTEGER,
                created_time TEXT,
                description TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _calculate_params_hash(self, preprocess_params: Dict) -> str:
        """计算预处理参数的哈希值"""
        params_str = json.dumps(preprocess_params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def save_database(self, name: str, folder_path: str, preprocess_params: Dict, 
                     peak_detection_params: Dict, library_spectra: Dict, description: str = "") -> str:
        """
        保存RRUFF库为数据库
        
        Args:
            name: 数据库名称
            folder_path: 原始文件夹路径
            preprocess_params: 预处理参数字典
            peak_detection_params: 峰值检测参数字典
            library_spectra: 光谱数据字典
            description: 数据库描述
            
        Returns:
            db_path: 数据库文件路径
        """
        params_hash = self._calculate_params_hash(preprocess_params)
        
        # 数据库文件名：name_params_hash.pkl
        db_filename = f"{name}_{params_hash[:8]}.pkl"
        db_path = os.path.join(self.db_dir, db_filename)
        
        # 保存数据
        db_data = {
            'name': name,
            'folder_path': folder_path,
            'params_hash': params_hash,
            'preprocess_params': preprocess_params,
            'peak_detection_params': peak_detection_params,
            'library_spectra': library_spectra,
            'spectra_count': len(library_spectra),
        }
        
        with open(db_path, 'wb') as f:
            pickle.dump(db_data, f)
        
        # 更新索引
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO databases 
            (name, folder_path, params_hash, preprocess_params, peak_detection_params, 
             spectra_count, created_time, description)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)
        ''', (
            name,
            folder_path,
            params_hash,
            json.dumps(preprocess_params),
            json.dumps(peak_detection_params),
            len(library_spectra),
            description
        ))
        conn.commit()
        conn.close()
        
        return db_path
    
    def load_database(self, name: str) -> Optional[Dict]:
        """
        加载数据库
        
        Args:
            name: 数据库名称
            
        Returns:
            db_data: 数据库数据字典，如果不存在则返回None
        """
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT params_hash FROM databases WHERE name = ?', (name,))
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return None
        
        params_hash = result[0]
        
        # 查找数据库文件
        db_filename = f"{name}_{params_hash[:8]}.pkl"
        db_path = os.path.join(self.db_dir, db_filename)
        
        if not os.path.exists(db_path):
            return None
        
        with open(db_path, 'rb') as f:
            return pickle.load(f)
    
    def find_database_by_params(self, preprocess_params: Dict) -> Optional[str]:
        """
        根据预处理参数自动查找匹配的数据库
        
        Args:
            preprocess_params: 预处理参数字典
            
        Returns:
            db_name: 匹配的数据库名称，如果没有则返回None
        """
        params_hash = self._calculate_params_hash(preprocess_params)
        
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM databases WHERE params_hash = ?', (params_hash,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        return None
    
    def list_databases(self) -> List[Dict]:
        """
        列出所有数据库
        
        Returns:
            databases: 数据库信息列表
        """
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name, folder_path, params_hash, preprocess_params, 
                   peak_detection_params, spectra_count, created_time, description
            FROM databases
            ORDER BY created_time DESC
        ''')
        results = cursor.fetchall()
        conn.close()
        
        databases = []
        for row in results:
            databases.append({
                'name': row[0],
                'folder_path': row[1],
                'params_hash': row[2],
                'preprocess_params': json.loads(row[3]),
                'peak_detection_params': json.loads(row[4]) if row[4] else {},
                'spectra_count': row[5],
                'created_time': row[6],
                'description': row[7] or ''
            })
        return databases
    
    def delete_database(self, name: str) -> bool:
        """
        删除数据库
        
        Args:
            name: 数据库名称
            
        Returns:
            success: 是否成功删除
        """
        conn = sqlite3.connect(self.index_db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT params_hash FROM databases WHERE name = ?', (name,))
        result = cursor.fetchone()
        
        if result:
            params_hash = result[0]
            db_filename = f"{name}_{params_hash[:8]}.pkl"
            db_path = os.path.join(self.db_dir, db_filename)
            
            # 删除文件
            if os.path.exists(db_path):
                os.remove(db_path)
            
            # 删除索引
            cursor.execute('DELETE FROM databases WHERE name = ?', (name,))
            conn.commit()
            conn.close()
            return True
        
        conn.close()
        return False

