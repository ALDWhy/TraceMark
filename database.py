"""
数据库操作模块 - 水印操作日志管理
"""
import sqlite3
import os
from datetime import datetime

# 数据库文件路径
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tracemark.db')

def init_db():
    """初始化数据库表"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建水印操作日志表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watermark_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT NOT NULL,
            filename TEXT,
            operation_type TEXT,
            embed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            strategy TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def log_watermark_operation(uid, filename, operation_type, ip_address=None, strategy=None):
    """记录水印操作日志"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO watermark_log (uid, filename, operation_type, ip_address, strategy)
        VALUES (?, ?, ?, ?, ?)
    ''', (uid, filename, operation_type, ip_address, strategy))
    
    conn.commit()
    conn.close()

def get_watermark_history(uid=None, limit=20):
    """查询水印操作历史"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if uid:
        cursor.execute('''
            SELECT * FROM watermark_log 
            WHERE uid LIKE ? 
            ORDER BY embed_time DESC 
            LIMIT ?
        ''', (f'%{uid}%', limit))
    else:
        cursor.execute('''
            SELECT * FROM watermark_log 
            ORDER BY embed_time DESC 
            LIMIT ?
        ''', (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [{
        'id': r[0],
        'uid': r[1],
        'filename': r[2],
        'operation_type': r[3],
        'embed_time': r[4],
        'ip_address': r[5],
        'strategy': r[6]
    } for r in results]

def get_watermark_stats(uid=None):
    """获取统计数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 总操作次数（整个系统所有操作）
    cursor.execute('SELECT COUNT(*) FROM watermark_log')
    total_operations = cursor.fetchone()[0]
    
    # 今日操作次数（使用 SQLite 的 DATE 函数处理时区）
    cursor.execute('SELECT COUNT(*) FROM watermark_log WHERE DATE(embed_time, "localtime") = DATE("now", "localtime")')
    today_operations = cursor.fetchone()[0]
    
    # 当前 UID 的嵌入次数
    if uid:
        cursor.execute('SELECT COUNT(*) FROM watermark_log WHERE operation_type = ? AND uid LIKE ?', ('embed', f'%{uid}%'))
        uid_embeds = cursor.fetchone()[0]
    else:
        cursor.execute('SELECT COUNT(*) FROM watermark_log WHERE operation_type = ?', ('embed',))
        uid_embeds = cursor.fetchone()[0]
    
    # 验证次数（整个系统）
    cursor.execute('SELECT COUNT(*) FROM watermark_log WHERE operation_type = ?', ('verify',))
    total_verifies = cursor.fetchone()[0]
    
    # 检测次数（整个系统）
    cursor.execute('SELECT COUNT(*) FROM watermark_log WHERE operation_type = ?', ('detect',))
    total_detects = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_operations': total_operations,  # 总操作次数（系统）
        'today_operations': today_operations,  # 今日操作次数
        'uid_embeds': uid_embeds,              # 当前UID的嵌入次数
        'total_verifies': total_verifies,      # 总验证次数
        'total_detects': total_detects         # 总检测次数
    }


def clear_watermark_history(uid=None):
    """
    清除操作历史
    
    Args:
        uid: 可选，指定要清除的UID。如果为None，则清除所有历史
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if uid:
        cursor.execute('DELETE FROM watermark_log WHERE uid LIKE ?', (f'%{uid}%',))
        deleted_count = cursor.rowcount
    else:
        cursor.execute('DELETE FROM watermark_log')
        deleted_count = cursor.rowcount
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="watermark_log"')
    
    conn.commit()
    conn.close()
    
    return deleted_count


# 初始化数据库
init_db()