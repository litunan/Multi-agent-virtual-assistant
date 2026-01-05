#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版多Agent数据分析系统 - 专业架构版
集成UserID、SessionID、ThreadID三层次架构
"""

from __future__ import annotations
import os
import asyncio
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Literal, Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph 
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from config.load_key import load_key

# 导入所有增强版Agent
# 使用 LlamaIndex 版本的 RAG Agent
from enhanced_rag_agent_llamaindex import enhanced_rag_agent
from enhanced_python_agent import enhanced_python_agent
from enhanced_amap_agent import enhanced_amap_agent
from safe_file_agent import safe_file_agent
from sql_agent import sql_agent
from terminal_command_agent import terminal_command_agent
from web_search_agent import web_search_agent

# 初始化模型 - 使用阿里云百炼 API
model = ChatOpenAI(
    api_key=load_key("aliyun-bailian"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)


# =============================================================================
# 专业记忆管理系统（三层次架构）
# =============================================================================

class ProfessionalMemoryManager:
    """专业记忆管理器 - 支持UserID、SessionID、ThreadID三层次架构"""

    def __init__(self, db_path: str = "professional_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """初始化专业数据库表"""
        cursor = self.conn.cursor()

        # 用户主表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')

        # 会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                session_context TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # 对话记忆表（短期记忆）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                agent_name TEXT,
                user_input TEXT,
                agent_response TEXT,
                context_summary TEXT,
                importance INTEGER DEFAULT 1,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        # 长期知识表（用户级别）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS long_term_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key_topic TEXT,
                information TEXT,
                category TEXT,
                importance INTEGER DEFAULT 1,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, key_topic),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # 用户偏好表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                preference_type TEXT NOT NULL,
                preference_value TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, preference_type),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')

        # 线程状态表（用于LangGraph状态管理）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS thread_states (
                thread_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                current_state TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_thread ON conversation_memory(session_id, thread_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_topic ON long_term_knowledge(user_id, key_topic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_prefs ON user_preferences(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_thread_session ON thread_states(session_id)')

        self.conn.commit()

    def create_user(self, user_id: str, username: str = None, metadata: Dict = None) -> bool:
        """创建新用户"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO users (user_id, username, metadata)
                VALUES (?, ?, ?)
            ''', (user_id, username, json.dumps(metadata) if metadata else None))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"创建用户失败: {e}")
            return False

    def create_session(self, session_id: str, user_id: str, context: str = "") -> bool:
        """创建新会话"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (session_id, user_id, session_context)
                VALUES (?, ?, ?)
            ''', (session_id, user_id, context))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"创建会话失败: {e}")
            return False

    def end_session(self, session_id: str):
        """结束会话"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET ended_at = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        ''', (session_id,))
        self.conn.commit()

    def add_conversation_memory(self, session_id: str, thread_id: str, agent_name: str,
                                user_input: str, agent_response: str,
                                context_summary: str = "", importance: int = 1, metadata: Dict = None):
        """添加对话记忆（短期记忆）"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversation_memory 
            (session_id, thread_id, agent_name, user_input, agent_response, context_summary, importance, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, thread_id, agent_name, user_input, agent_response,
              context_summary, importance, json.dumps(metadata) if metadata else None))
        self.conn.commit()

    def get_recent_memory(self, session_id: str, thread_id: str = None, limit: int = 10) -> List[Dict]:
        """获取最近的短期记忆"""
        cursor = self.conn.cursor()

        if thread_id:
            cursor.execute('''
                SELECT thread_id, agent_name, user_input, agent_response, context_summary, timestamp, importance
                FROM conversation_memory 
                WHERE session_id = ? AND thread_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, thread_id, limit))
        else:
            cursor.execute('''
                SELECT thread_id, agent_name, user_input, agent_response, context_summary, timestamp, importance
                FROM conversation_memory 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, limit))

        rows = cursor.fetchall()
        return [
            {
                "thread_id": row[0],
                "agent": row[1],
                "user_input": row[2],
                "agent_response": row[3],
                "context": row[4],
                "timestamp": row[5],
                "importance": row[6]
            }
            for row in rows
        ]

    def get_session_threads(self, session_id: str) -> List[str]:
        """获取会话中的所有线程ID"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT thread_id
            FROM conversation_memory 
            WHERE session_id = ?
            ORDER BY timestamp DESC
        ''', (session_id,))

        rows = cursor.fetchall()
        return [row[0] for row in rows]

    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户的所有会话"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT session_id, started_at, ended_at, session_context
            FROM sessions 
            WHERE user_id = ?
            ORDER BY started_at DESC 
            LIMIT ?
        ''', (user_id, limit))

        rows = cursor.fetchall()
        return [
            {
                "session_id": row[0],
                "started_at": row[1],
                "ended_at": row[2],
                "session_context": row[3]
            }
            for row in rows
        ]

    def add_long_term_knowledge(self, user_id: str, key_topic: str, information: str,
                                category: str = "general", importance: int = 1):
        """添加长期知识（用户级别）"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO long_term_knowledge 
            (user_id, key_topic, information, category, importance, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 
                   COALESCE((SELECT access_count FROM long_term_knowledge WHERE user_id = ? AND key_topic = ?), 0) + 1)
        ''', (user_id, key_topic, information, category, importance, user_id, key_topic))
        self.conn.commit()

    def search_long_term_knowledge(self, user_id: str, query: str, category: str = None, limit: int = 5) -> List[Dict]:
        """搜索用户的长期知识"""
        cursor = self.conn.cursor()

        if category:
            cursor.execute('''
                SELECT key_topic, information, category, importance, last_accessed
                FROM long_term_knowledge 
                WHERE user_id = ? AND (key_topic LIKE ? OR information LIKE ?) AND category = ?
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            ''', (user_id, f'%{query}%', f'%{query}%', category, limit))
        else:
            cursor.execute('''
                SELECT key_topic, information, category, importance, last_accessed
                FROM long_term_knowledge 
                WHERE user_id = ? AND (key_topic LIKE ? OR information LIKE ?)
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            ''', (user_id, f'%{query}%', f'%{query}%', limit))

        rows = cursor.fetchall()
        return [
            {
                "topic": row[0],
                "information": row[1],
                "category": row[2],
                "importance": row[3],
                "last_accessed": row[4]
            }
            for row in rows
        ]

    def save_user_preference(self, user_id: str, preference_type: str, preference_value: str):
        """保存用户偏好"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (user_id, preference_type, preference_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, preference_type, preference_value))
        self.conn.commit()

    def get_user_preferences(self, user_id: str) -> Dict[str, str]:
        """获取用户偏好"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT preference_type, preference_value
            FROM user_preferences 
            WHERE user_id = ?
        ''', (user_id,))

        rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    def update_thread_state(self, thread_id: str, session_id: str, state: str):
        """更新线程状态"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO thread_states 
            (thread_id, session_id, current_state, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (thread_id, session_id, state))
        self.conn.commit()

    def get_thread_state(self, thread_id: str) -> Optional[str]:
        """获取线程状态"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT current_state
            FROM thread_states 
            WHERE thread_id = ?
        ''', (thread_id,))

        row = cursor.fetchone()
        return row[0] if row else None

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        cursor = self.conn.cursor()

        # 用户基本信息
        cursor.execute('SELECT username, created_at, last_active FROM users WHERE user_id = ?', (user_id,))
        user_info = cursor.fetchone()

        # 会话统计
        cursor.execute('SELECT COUNT(*) FROM sessions WHERE user_id = ?', (user_id,))
        session_count = cursor.fetchone()[0]

        # 对话记忆统计
        cursor.execute('''
            SELECT COUNT(*) 
            FROM conversation_memory cm
            JOIN sessions s ON cm.session_id = s.session_id
            WHERE s.user_id = ?
        ''', (user_id,))
        memory_count = cursor.fetchone()[0]

        # 长期知识统计
        cursor.execute('SELECT COUNT(*) FROM long_term_knowledge WHERE user_id = ?', (user_id,))
        knowledge_count = cursor.fetchone()[0]

        return {
            "username": user_info[0] if user_info else None,
            "created_at": user_info[1] if user_info else None,
            "last_active": user_info[2] if user_info else None,
            "session_count": session_count,
            "memory_count": memory_count,
            "knowledge_count": knowledge_count
        }

    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')

        # 清理旧的对话记忆（保留重要记忆）
        cursor.execute('''
            DELETE FROM conversation_memory 
            WHERE timestamp < ? AND importance < 3
        ''', (cutoff_date,))

        # 清理已结束的旧会话
        cursor.execute('''
            DELETE FROM sessions 
            WHERE ended_at IS NOT NULL AND ended_at < ?
        ''', (cutoff_date,))

        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


# 全局专业记忆管理器实例
professional_memory = ProfessionalMemoryManager()


# =============================================================================
# 专业会话管理器
# =============================================================================

class SessionManager:
    """专业会话管理器"""

    def __init__(self, memory_manager: ProfessionalMemoryManager):
        self.memory_manager = memory_manager
        self.active_sessions: Dict[str, Dict] = {}

    def create_user_session(self, user_id: str, username: str = None,
                            session_context: str = "") -> Dict[str, str]:
        """创建用户会话"""
        # 确保用户存在
        self.memory_manager.create_user(user_id, username)

        # 生成会话ID和线程ID
        session_id = f"{user_id}_{int(datetime.now().timestamp())}"
        thread_id = f"{session_id}_thread_{uuid.uuid4().hex[:8]}"

        # 创建会话
        self.memory_manager.create_session(session_id, user_id, session_context)

        # 记录活跃会话
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "thread_id": thread_id,
            "created_at": datetime.now(),
            "context": session_context
        }

        return {
            "user_id": user_id,
            "session_id": session_id,
            "thread_id": thread_id
        }

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        return self.active_sessions.get(session_id)

    def create_thread(self, session_id: str) -> str:
        """为会话创建新线程"""
        if session_id not in self.active_sessions:
            raise ValueError(f"会话不存在: {session_id}")

        thread_id = f"{session_id}_thread_{uuid.uuid4().hex[:8]}"
        self.active_sessions[session_id]["thread_id"] = thread_id

        return thread_id

    def end_session(self, session_id: str):
        """结束会话"""
        if session_id in self.active_sessions:
            self.memory_manager.end_session(session_id)
            del self.active_sessions[session_id]


# =============================================================================
# 增强版Supervisor（集成专业记忆系统）
# =============================================================================

PROFESSIONAL_SUPERVISOR_PROMPT = """你是一个顶级的AI项目总监，名字叫小楷，负责管理一个由七位小弟组成的AI团队。
你的职责是根据用户的需求，智能地将任务分配给最合适的专家处理。

**🎯 你的团队成员档案：**

1. **`enhanced_rag_agent` (抗癌肽研究专家助手)**
   - **能力**: 深度理解用户的提问，具备多轮对话记忆和智能问答能力
   - **调用时机**:
     - 用户提出抗癌肽相关问题时
     - 需要专业术语解释
     - 需要上下文关联分析的复杂问题

2. **`enhanced_python_agent` (高级Python数据科学专家)**
   - **能力**: 强大的Python代码执行、数据分析、机器学习和高级可视化
   - **调用时机**:
     - Python代码编写和执行
     - 数据科学分析和统计计算
     - 机器学习模型构建和评估
     - 高级数据可视化和图表生成
     - 数据处理和清洗任务

3. **`enhanced_amap_agent` (全功能地理位置服务专家)**
   - **能力**: 完整的地理位置服务，包括天气、导航、区域分析和行程规划
   - **调用时机**:
     - 天气查询和预报分析
     - 地理编码和位置转换
     - POI搜索和区域设施分析
     - 路径规划和交通分析
     - 批量地理数据处理

4. **`safe_file_agent` (安全文件管理专家)**
   - **能力**: 在安全沙盒环境中进行文件操作，包括读写、管理和组织
   - **调用时机**:
     - 文件创建、读取、修改、删除
     - 目录管理和文件组织
     - 文档编辑和内容处理
     - 配置文件管理
     - 日志文件分析

5. **`sql_agent` (数据库操作专家)**
   - **能力**: 数据库查询、数据提取和数据库管理
   - **调用时机**:
     - SQL查询和数据库操作
     - 数据提取和导出
     - 数据库结构分析
     - 为其他Agent准备数据

6. **`terminal_command_agent` (终端命令执行和文件操作专家)**
   - **能力**: 安全的终端命令执行、完整的文件系统操作和系统管理
   - **调用时机**:
     - 终端命令执行和系统操作
     - 文件的增删改查操作
     - 目录管理和文件组织
     - 系统信息查询和监控
     - 开发工具调用和脚本执行
     - 网络工具使用和数据下载

7. **`web_search_agent` (智能网络搜索检索专家)**
   - **能力**: 基于Tavily API的实时网络搜索、新闻检索、金融信息查询和URL内容提取
   - **调用时机**:
     - 需要最新网络信息和实时资讯
     - 新闻事件查询和时事分析
     - 金融市场信息和股票数据检索
     - 技术文档和学术资料搜索
     - 从指定URL提取和分析内容
     - 竞品分析和市场调研
     - 验证信息的准确性和时效性

**🧠 专业记忆系统说明：**

你有一个完整的专业记忆系统，包括：
- **用户级别**: 长期记忆和用户偏好（基于UserID）
- **会话级别**: 单次对话会话（基于SessionID）
- **线程级别**: 对话线程状态管理（基于ThreadID）

在决策时，请考虑：
1. 用户的历史偏好和长期知识
2. 当前会话的上下文
3. 线程的当前状态

**🔄 工作流程原则：**

- **智能路由**: 准确识别用户需求，选择最适合的专家
- **协作配合**: 当任务需要多个专家时，合理安排协作顺序
- **依赖管理**: 确保数据依赖关系得到满足
- **结果整合**: 将多个专家的结果有机整合，提供完整解决方案
- **用户体验**: 保持流畅的交互体验，及时反馈处理进度
- **记忆利用**: 充分利用三层次记忆系统提供连贯、个性化的服务

**📋 决策过程：**

1. **需求分析**: 仔细分析用户请求的类型和复杂度
2. **记忆检索**: 检查相关记忆（用户、会话、线程级别）
3. **专家选择**: 选择最匹配的专家或专家组合
4. **任务分解**: 将复杂任务分解为专家可处理的子任务
5. **执行监控**: 跟踪任务执行进度和质量
6. **结果验证**: 确保输出满足用户需求
7. **记忆更新**: 将重要信息保存到相应的记忆系统中
8. **输出结果**: 当所有子任务完成时，输出你整合后的专家的答案,内容要全面细节，如果有数据类的结果要全部保留

**🎯 可选择的下一步行动：**
`['enhanced_rag_agent', 'enhanced_python_agent', 'enhanced_amap_agent', 'safe_file_agent', 'sql_agent', 'terminal_command_agent', 'web_search_agent', 'FINISH']`

请根据用户需求，选择最合适的专家来处理任务！

"""

# 创建增强版Supervisor（集成专业记忆系统）- 使用阿里云百炼 API
supervisor_model = ChatOpenAI(
    api_key=load_key("aliyun-bailian"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0,
)

professional_supervisor = create_supervisor(
    model=supervisor_model,
    agents=[enhanced_rag_agent, enhanced_python_agent, enhanced_amap_agent, safe_file_agent, sql_agent,
            terminal_command_agent, web_search_agent],
    prompt=PROFESSIONAL_SUPERVISOR_PROMPT,
    add_handoff_back_messages=True
)

# 编译supervisor
supervisor_graph = professional_supervisor.compile()


# =============================================================================
# 专业记忆增强的查询接口
# =============================================================================

class ProfessionalSystemWithMemory:
    """专业版多Agent系统（三层次架构）"""

    def __init__(self):
        self.supervisor = supervisor_graph
        self.memory_manager = professional_memory
        self.session_manager = SessionManager(professional_memory)

    def _extract_key_topics(self, text: str) -> List[str]:
        """从文本中提取关键主题"""
        words = text.lower().split()
        important_words = [word for word in words if len(word) > 3 and word not in
                           ['what', 'when', 'where', 'which', 'how', 'why', 'this', 'that', 'with', 'from']]
        return important_words[:5]

    def _build_memory_context(self, user_id: str, session_id: str, thread_id: str, user_input: str) -> str:
        """构建专业记忆上下文"""
        context_parts = []

        # 获取会话级别的短期记忆
        short_term_memories = self.memory_manager.get_recent_memory(session_id, thread_id, limit=5)
        if short_term_memories:
            context_parts.append("## 当前会话记忆:")
            for i, memory in enumerate(short_term_memories[::-1]):
                context_parts.append(f"{i + 1}. [{memory['agent']}] 用户: {memory['user_input'][:100]}...")

        # 获取用户级别的长期知识
        key_topics = self._extract_key_topics(user_input)
        if key_topics:
            related_knowledge = []
            for topic in key_topics:
                knowledge = self.memory_manager.search_long_term_knowledge(user_id, topic, limit=2)
                related_knowledge.extend(knowledge)

            if related_knowledge:
                context_parts.append("\n## 相关知识:")
                for knowledge in related_knowledge[:3]:
                    context_parts.append(f"- {knowledge['topic']}: {knowledge['information'][:100]}...")

        # 获取用户偏好
        user_prefs = self.memory_manager.get_user_preferences(user_id)
        if user_prefs:
            context_parts.append("\n## 用户偏好:")
            for pref_type, pref_value in user_prefs.items():
                context_parts.append(f"- {pref_type}: {pref_value}")

        # 获取线程状态
        thread_state = self.memory_manager.get_thread_state(thread_id)
        if thread_state:
            context_parts.append(f"\n## 当前线程状态: {thread_state}")

        return "\n".join(context_parts) if context_parts else "暂无相关记忆上下文"

    async def query(self, user_input: str, user_id: str, session_id: str = None,
                    thread_id: str = None, username: str = None) -> str:
        """执行专业记忆查询"""
        # 如果没有提供session_id，创建新会话
        if session_id is None:
            session_info = self.session_manager.create_user_session(
                user_id, username, f"用户查询: {user_input[:50]}..."
            )
            session_id = session_info["session_id"]
            thread_id = session_info["thread_id"]
        else:
            # 如果没有提供thread_id，使用会话的主线程或创建新线程
            if thread_id is None:
                session_info = self.session_manager.get_session_info(session_id)
                if session_info:
                    thread_id = session_info["thread_id"]
                else:
                    thread_id = f"{session_id}_main_thread"

        try:
            # 构建专业记忆增强的输入
            memory_context = self._build_memory_context(user_id, session_id, thread_id, user_input)

            enhanced_input = f"""
            用户身份: {user_id} ({username or '匿名用户'})
            会话ID: {session_id}
            线程ID: {thread_id}
            
            记忆上下文:
            {memory_context}
            
            当前查询: {user_input}
            
            请基于以上用户身份和记忆上下文，提供最合适的响应。
            """

            # 更新线程状态
            self.memory_manager.update_thread_state(thread_id, session_id, "processing_query")

            # 执行supervisor查询
            result = await self.supervisor.ainvoke({
                "messages": [{"role": "user", "content": enhanced_input}]
            })

            # 提取响应内容
            if result and "messages" in result and result["messages"]:
                response_content = result["messages"][-1].content
            else:
                response_content = "抱歉，我没有收到有效的响应。"

            # 更新线程状态
            self.memory_manager.update_thread_state(thread_id, session_id, "completed")

            # 保存到专业记忆系统
            self._save_to_professional_memory(user_id, session_id, thread_id, user_input, response_content,
                                              enhanced_input)

            return response_content

        except Exception as e:
            error_msg = f"❌ 系统处理失败: {str(e)}"
            # 即使出错也记录到记忆
            self.memory_manager.add_conversation_memory(
                session_id=session_id,
                thread_id=thread_id,
                agent_name="system",
                user_input=user_input,
                agent_response=error_msg,
                context_summary=f"System error: {str(e)}",
                importance=1
            )
            return error_msg

    def _save_to_professional_memory(self, user_id: str, session_id: str, thread_id: str,
                                     user_input: str, response: str, enhanced_input: str):
        """保存对话到专业记忆系统"""
        # 确定重要性级别
        importance = 1
        if any(keyword in user_input.lower() for keyword in
               ['重要', '记住', '偏好', '喜欢', '不喜欢', '设置']):
            importance = 3
        elif any(keyword in user_input.lower() for keyword in
                 ['如何', '教程', '步骤', '方法', '解释']):
            importance = 2

        # 保存到会话记忆
        self.memory_manager.add_conversation_memory(
            session_id=session_id,
            thread_id=thread_id,
            agent_name="AI项目总监-小楷",
            user_input=user_input,
            agent_response=response,
            context_summary=f"User query about {self._extract_key_topics(user_input)}",
            importance=importance
        )

        # 如果很重要，保存到用户长期知识
        if importance >= 3:
            key_topics = self._extract_key_topics(user_input)
            for topic in key_topics[:2]:
                self.memory_manager.add_long_term_knowledge(
                    user_id=user_id,
                    key_topic=topic,
                    information=f"用户查询: {user_input}\n系统响应: {response[:200]}...",
                    category="user_preference",
                    importance=importance
                )

        # 检测用户偏好设置
        self._detect_user_preferences(user_id, user_input, response)

        # 更新用户最后活跃时间
        cursor = self.memory_manager.conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET last_active = CURRENT_TIMESTAMP 
            WHERE user_id = ?
        ''', (user_id,))
        self.memory_manager.conn.commit()

    def _detect_user_preferences(self, user_id: str, user_input: str, response: str):
        """检测并保存用户偏好"""
        user_input_lower = user_input.lower()

        # 检测语言偏好
        if any(word in user_input_lower for word in ['中文', '英文', '语言']):
            if '英文' in user_input_lower or 'english' in user_input_lower:
                self.memory_manager.save_user_preference(user_id, "language", "english")
            else:
                self.memory_manager.save_user_preference(user_id, "language", "chinese")

        # 检测详细程度偏好
        if '详细' in user_input_lower or '详细点' in user_input_lower:
            self.memory_manager.save_user_preference(user_id, "detail_level", "detailed")
        elif '简洁' in user_input_lower or '简单' in user_input_lower:
            self.memory_manager.save_user_preference(user_id, "detail_level", "concise")

        # 检测主题偏好
        preferred_topics = []
        for topic in ['python', '数据', '地图', '天气', '文件', '搜索', '数据库']:
            if topic in user_input_lower:
                preferred_topics.append(topic)

        if preferred_topics:
            self.memory_manager.save_user_preference(user_id, "preferred_topics", ",".join(preferred_topics))

    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """获取用户仪表板信息"""
        # 用户基本信息
        user_stats = self.memory_manager.get_user_stats(user_id)

        # 用户会话列表
        user_sessions = self.memory_manager.get_user_sessions(user_id, limit=5)

        # 用户偏好
        user_preferences = self.memory_manager.get_user_preferences(user_id)

        return {
            "user_info": user_stats,
            "recent_sessions": user_sessions,
            "preferences": user_preferences
        }

    def create_new_thread(self, session_id: str) -> str:
        """为会话创建新线程"""
        return self.session_manager.create_thread(session_id)

    def end_user_session(self, session_id: str):
        """结束用户会话"""
        self.session_manager.end_session(session_id)


# 创建专业系统实例
professional_system = ProfessionalSystemWithMemory()


# =============================================================================
# 简化的专业查询接口
# =============================================================================

async def professional_system_query(user_query: str, user_id: str,
                                    session_id: str = None, thread_id: str = None,
                                    username: str = None) -> str:
    """
    专业版系统查询接口（三层次架构）
    """
    # 处理特殊命令
    if user_query.lower() in ['仪表板', 'dashboard']:
        dashboard = professional_system.get_user_dashboard(user_id)
        return f"""
用户仪表板 - {user_id}
====================
用户信息:
  用户名: {dashboard['user_info'].get('username', '未设置')}
  创建时间: {dashboard['user_info'].get('created_at', '未知')}
  最后活跃: {dashboard['user_info'].get('last_active', '未知')}
  会话数量: {dashboard['user_info'].get('session_count', 0)}
  记忆数量: {dashboard['user_info'].get('memory_count', 0)}
  知识数量: {dashboard['user_info'].get('knowledge_count', 0)}

最近会话:
{chr(10).join([f"  - {s['session_id']} ({s['started_at']})" for s in dashboard['recent_sessions']])}

用户偏好:
{chr(10).join([f"  - {k}: {v}" for k, v in dashboard['preferences'].items()])}
====================
        """

    elif user_query.lower() in ['清理数据', 'cleanup data']:
        professional_memory.cleanup_old_data(30)
        return "✅ 已清理30天前的非重要数据"

    elif user_query.lower() in ['新线程', 'new thread'] and session_id:
        new_thread_id = professional_system.create_new_thread(session_id)
        return f"✅ 已创建新线程: {new_thread_id}"

    elif user_query.lower() in ['帮助', 'help']:
        return get_professional_system_info()

    # 正常查询
    return await professional_system.query(user_query, user_id, session_id, thread_id, username)


def get_professional_system_info() -> str:
    """获取专业版系统信息"""
    info = """
🚀 专业版多Agent数据分析系统（三层次架构）
=====================================

📊 **系统架构**:
   • UserID - 用户身份标识（长期记忆和偏好）
   • SessionID - 会话标识（短期记忆和上下文）
   • ThreadID - 线程标识（状态管理和对话流）

🧠 **记忆层次**:
   • 用户级别: 长期知识、用户偏好、历史统计
   • 会话级别: 短期对话记忆、会话上下文
   • 线程级别: 对话状态、执行进度

🔧 **管理功能**:
   • 用户仪表板（输入"仪表板"查看）
   • 多线程对话支持
   • 会话生命周期管理
   • 数据自动清理和维护

💡 **特殊命令**:
   • "仪表板" - 显示用户统计和偏好
   • "清理数据" - 清理旧数据
   • "帮助" - 显示此帮助信息

=====================================
"""
    return info


# =============================================================================
# 演示和测试功能
# =============================================================================

async def run_professional_demo():
    """运行专业版系统演示"""
    print(get_professional_system_info())

    # 创建测试用户
    user_id = "test_user_001"
    username = "演示用户"

    print(f"🧪 当前测试用户: {user_id} ({username})")

    # 初始会话
    current_session = "test_user_001_1761732028"
    current_thread = None
    print(f"🧪 当前测试会话: {current_session} ")
    while True:
        try:
            query = input("\n请输入您的查询（输入 'exit' 退出）：")
        except KeyboardInterrupt:
            print("\n\n⏹️ 系统演示被中断")
            break

        if query.lower() == "exit":
            if current_session:
                professional_system.end_user_session(current_session)
            print("感谢使用专业版多Agent系统，程序已退出。")
            break

        try:
            response = await professional_system_query(
                query, user_id, current_session, current_thread, username
            )

            # 更新当前会话和线程信息
            if not current_session:
                # 从响应中提取会话信息（在实际系统中应该从返回对象中获取）
                current_session = f"{user_id}_{int(datetime.now().timestamp())}"
                current_thread = f"{current_session}_main_thread"

            print(f"\n🤖 系统回复:\n{response}")

        except Exception as e:
            print(f"❌ 发生错误：{str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("🚀 启动专业版多Agent数据分析系统（三层次架构）...")

    # 显示系统信息
    print(get_professional_system_info())

    # 运行专业演示
    try:
        asyncio.run(run_professional_demo())
    except KeyboardInterrupt:
        print("\n⏹️ 系统演示被中断")
    except Exception as e:
        print(f"\n❌ 系统启动失败: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 关闭专业记忆管理器
        professional_memory.close()
        print("✅ 专业记忆系统已安全关闭")