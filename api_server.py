#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多Agent系统前端API服务 - 修复版
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
import os

# 导入之前的专业系统
try:
    from enhanced_data_agent1 import ProfessionalSystemWithMemory, professional_system_query
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 系统将在模拟模式下运行")


    # 创建一个简单的模拟类用于测试
    class ProfessionalSystemWithMemory:
        def __init__(self):
            self.memory_manager = None
            self.session_manager = None


    async def professional_system_query(message, user_id, session_id=None, thread_id=None):
        return f"测试回复: {message}"

# 创建FastAPI应用
app = FastAPI(title="多Agent系统API", description="多Agent数据分析系统前端API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化系统
professional_system = ProfessionalSystemWithMemory()


# 数据模型
class UserLoginRequest(BaseModel):
    user_id: str
    username: Optional[str] = None


class UserLoginResponse(BaseModel):
    user_id: str
    username: Optional[str]
    status: str


class CreateSessionRequest(BaseModel):
    session_context: str = "新会话"


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str


class SendMessageRequest(BaseModel):
    message: str


class SendMessageResponse(BaseModel):
    response: str
    status: str


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    started_at: str
    ended_at: Optional[str]
    session_context: str


class ThreadInfo(BaseModel):
    thread_id: str
    session_id: str
    created_at: str


class MessageInfo(BaseModel):
    message_id: str
    sender: str
    content: str
    timestamp: str
    agent_name: Optional[str] = None


class UserDashboardResponse(BaseModel):
    user_info: Dict[str, Any]
    recent_sessions: List[SessionInfo]
    preferences: Dict[str, str]


# API路由
@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "多Agent系统API"}


@app.post("/api/user/login", response_model=UserLoginResponse)
async def user_login(request: UserLoginRequest):
    """用户登录/注册"""
    try:
        print(f"用户登录请求: user_id={request.user_id}, username={request.username}")

        # 检查系统是否初始化
        if professional_system.memory_manager is None:
            return UserLoginResponse(
                user_id=request.user_id,
                username=request.username,
                status="success"
            )

        # 确保用户存在
        professional_system.memory_manager.create_user(request.user_id, request.username)

        # 如果提供了用户名，更新用户偏好
        if request.username:
            professional_system.memory_manager.save_user_preference(
                request.user_id, "username", request.username
            )

        return UserLoginResponse(
            user_id=request.user_id,
            username=request.username,
            status="success"
        )
    except Exception as e:
        print(f"用户登录错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"用户登录失败: {str(e)}")


@app.get("/api/user/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    """获取用户的所有会话"""
    try:
        # 检查系统是否初始化
        if professional_system.memory_manager is None:
            return {"sessions": []}

        sessions = professional_system.memory_manager.get_user_sessions(user_id, limit=20)
        print(sessions[0])
        return {"sessions": sessions}
    except Exception as e:
        print(f"获取用户会话错误: {str(e)}")
        return {"sessions": []}  # 返回空列表而不是抛出异常


@app.post("/api/user/{user_id}/sessions", response_model=CreateSessionResponse)
async def create_user_session(user_id: str, request: CreateSessionRequest):
    """创建新会话"""
    try:
        print(f"创建会话: user_id={user_id}, context={request.session_context}")

        # 检查系统是否初始化
        if professional_system.session_manager is None:
            # 返回模拟会话ID
            import time
            session_id = f"{user_id}_session_{int(time.time())}"
            return CreateSessionResponse(
                session_id=session_id,
                status="success"
            )

        # 创建会话
        session_info = professional_system.session_manager.create_user_session(
            user_id,
            None,  # 用户名可以从用户偏好中获取
            request.session_context
        )

        return CreateSessionResponse(
            session_id=session_info["session_id"],
            status="success"
        )
    except Exception as e:
        print(f"创建会话错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@app.get("/api/session/{session_id}/threads")
async def get_session_threads(session_id: str):
    """获取会话的所有线程"""
    try:
        # 检查系统是否初始化
        if professional_system.memory_manager is None:
            # 返回默认线程
            return {"threads": [{
                "thread_id": f"{session_id}_main_thread",
                "session_id": session_id,
                "created_at": "2024-01-01 00:00:00"
            }]}

        # 获取会话的所有线程
        threads = professional_system.memory_manager.get_session_threads(session_id)

        thread_infos = []
        for thread_id in threads:
            thread_infos.append({
                "thread_id": thread_id,
                "session_id": session_id,
                "created_at": "2024-01-01 00:00:00"  # 实际应该从数据库获取
            })

        # 如果没有线程，创建一个默认线程
        if not thread_infos:
            thread_infos.append({
                "thread_id": f"{session_id}_main_thread",
                "session_id": session_id,
                "created_at": "2024-01-01 00:00:00"
            })

        return {"threads": thread_infos}
    except Exception as e:
        print(f"获取线程列表错误: {str(e)}")
        # 返回默认线程而不是抛出异常
        return {"threads": [{
            "thread_id": f"{session_id}_main_thread",
            "session_id": session_id,
            "created_at": "2024-01-01 00:00:00"
        }]}


@app.get("/api/session/{session_id}/thread/{thread_id}/messages")
async def get_thread_messages(session_id: str, thread_id: str):
    """获取线程的消息历史 - 简洁修复版"""
    try:
        # 获取对话记忆
        memories = professional_system.memory_manager.get_recent_memory(
            session_id, thread_id, limit=50
        )

        messages = []
        for memory in memories:
            # 修复逻辑：每条记忆都包含用户输入和系统回复

            # 用户消息
            messages.append({
                "message_id": f"user_{hash(str(memory.get('timestamp', '')) + memory.get('user_input', ''))}",
                "sender": "user",
                "content": memory.get("user_input", ""),
                "timestamp": memory.get("timestamp", ""),
                "agent_name": None
            })

            # 系统回复消息（如果有）
            agent_response = memory.get("agent_response", "")
            if agent_response and agent_response.strip():
                messages.append({
                    "message_id": f"agent_{hash(str(memory.get('timestamp', '')) + agent_response)}",
                    "sender": "agent",
                    "content": agent_response,
                    "timestamp": memory.get("timestamp", ""),
                    "agent_name": memory.get("agent", "system")
                })

        # 按时间戳排序
        messages.sort(key=lambda x: x.get("timestamp", ""))

        return {"messages": messages}
    except Exception as e:
        print(f"获取消息历史错误: {str(e)}")
        return {"messages": []}


@app.post("/api/user/{user_id}/session/{session_id}/thread/{thread_id}/message",
          response_model=SendMessageResponse)
async def send_message(user_id: str, session_id: str, thread_id: str, request: SendMessageRequest):
    """发送消息到系统并获取回复"""
    try:
        print(f"发送消息: user={user_id}, session={session_id}, thread={thread_id}, message={request.message}")

        # 使用专业系统处理查询
        response = await professional_system_query(
            request.message, user_id, session_id, thread_id
        )

        return SendMessageResponse(
            response=response,
            status="success"
        )
    except Exception as e:
        print(f"处理消息错误: {str(e)}")
        # 返回模拟回复而不是抛出异常
        return SendMessageResponse(
            response=f"系统暂时无法处理您的请求: {str(e)}",
            status="success"
        )


@app.get("/api/user/{user_id}/dashboard")
async def get_user_dashboard(user_id: str):
    """获取用户仪表板信息"""
    try:
        # 检查系统是否初始化
        if hasattr(professional_system, 'get_user_dashboard') and professional_system.get_user_dashboard is None:
            return {
                "user_info": {
                    "username": "测试用户",
                    "session_count": 0,
                    "memory_count": 0,
                    "knowledge_count": 0
                },
                "recent_sessions": [],
                "preferences": {}
            }

        dashboard = professional_system.get_user_dashboard(user_id)
        return dashboard
    except Exception as e:
        print(f"获取仪表板错误: {str(e)}")
        return {
            "user_info": {
                "username": "测试用户",
                "session_count": 0,
                "memory_count": 0,
                "knowledge_count": 0
            },
            "recent_sessions": [],
            "preferences": {}
        }


# 添加favicon端点，避免404错误
@app.get("/favicon.ico")
async def get_favicon():
    return {"message": "No favicon"}


# 静态文件服务 - 提供前端页面
@app.get("/")
async def serve_frontend():
    return FileResponse("webUI.html")


# 启动服务
if __name__ == "__main__":
    print("🚀 启动多Agent系统API服务...")
    print("📝 访问地址: http://127.0.0.1:8020")
    print("💡 如果遇到导入错误，系统将使用模拟模式运行")

    try:
        uvicorn.run(app, host="127.0.0.1", port=8020, log_level="info")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请检查端口8020是否被占用")