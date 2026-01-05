#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全文件管理Agent
严格限制在指定工作目录内的文件操作
包含多重安全检查和权限控制

⚠️ 安全特性：
1. 沙盒环境 - 仅在指定目录内操作
2. 路径验证 - 防止路径遍历攻击
3. 操作白名单 - 只允许安全的文件操作
4. 文件类型检查 - 限制可操作的文件类型
5. 大小限制 - 防止过大文件操作
"""

import os
import json
import shutil
import pathlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv 
load_dotenv(override=True)
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config.load_key import load_key

# 初始化模型 - 使用阿里云百炼 API
model = ChatOpenAI(
    api_key=load_key("aliyun-bailian"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# =============================================================================
# 安全配置
# =============================================================================

# 工作目录沙盒 - 所有操作仅限在此目录内
SANDBOX_DIR = "workspace"

# 桌面目录 - 用于特殊文件输出
DESKTOP_DIR = "/Users/anthony/Desktop"

# 允许的文件扩展名（白名单）
ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.json', '.csv', '.log', '.py', '.js', '.html', '.css',
    '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf'
}

# 禁止的文件名模式
FORBIDDEN_PATTERNS = {
    'passwd', 'shadow', 'hosts', 'fstab', 'sudoers', 'crontab',
    '.ssh', '.env', 'id_rsa', 'private_key', 'secret'
}

# 文件大小限制（MB）
MAX_FILE_SIZE_MB = 10

# 禁止的系统目录
SYSTEM_DIRS = {
    '/etc', '/var', '/usr', '/bin', '/sbin', '/root', '/home',
    '/sys', '/proc', '/dev', '/tmp', '/boot'
}

def ensure_sandbox():
    """确保沙盒目录存在"""
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    
    # 创建示例文件结构
    subdirs = ['documents', 'logs', 'data', 'temp']
    for subdir in subdirs:
        os.makedirs(os.path.join(SANDBOX_DIR, subdir), exist_ok=True)
    
    # 创建README文件
    readme_path = os.path.join(SANDBOX_DIR, 'README.md')
    if not os.path.exists(readme_path):
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("""# 安全文件管理工作区

这是一个安全的文件管理工作区，所有文件操作仅限在此目录内。

## 目录结构
- `documents/` - 文档文件
- `logs/` - 日志文件  
- `data/` - 数据文件
- `temp/` - 临时文件

## 安全限制
- 只能操作指定扩展名的文件
- 文件大小限制为10MB
- 禁止访问系统目录
- 所有操作都有详细日志记录
""")

def validate_path(file_path: str) -> tuple[bool, str]:
    """验证文件路径安全性"""
    try:
        # 转换为绝对路径
        abs_path = os.path.abspath(file_path)
        sandbox_abs = os.path.abspath(SANDBOX_DIR)
        
        # 检查是否在沙盒目录内
        if not abs_path.startswith(sandbox_abs):
            return False, f"路径超出安全工作区范围: {file_path}"
        
        # 检查路径遍历攻击
        if '..' in file_path or file_path.startswith('/'):
            return False, f"检测到不安全的路径模式: {file_path}"
        
        # 检查系统目录
        for sys_dir in SYSTEM_DIRS:
            if abs_path.startswith(sys_dir):
                return False, f"禁止访问系统目录: {file_path}"
        
        # 检查禁止的文件名模式（检查完整路径，不仅仅是文件名）
        path_lower = file_path.lower()
        filename = os.path.basename(file_path).lower()
        for pattern in FORBIDDEN_PATTERNS:
            # 检查文件名是否包含禁止模式
            if pattern in filename:
                return False, f"文件名包含禁止模式: {pattern}"
            # 检查完整路径是否包含禁止模式（如 .ssh 目录）
            if pattern in path_lower:
                return False, f"路径包含禁止模式: {pattern}"
        
        return True, ""
        
    except Exception as e:
        return False, f"路径验证失败: {str(e)}"

def validate_file_extension(file_path: str) -> tuple[bool, str]:
    """验证文件扩展名"""
    ext = pathlib.Path(file_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支持的文件类型: {ext}。允许的类型: {', '.join(ALLOWED_EXTENSIONS)}"
    return True, ""

def get_safe_file_path(relative_path: str) -> str:
    """获取安全的文件路径"""
    return os.path.join(SANDBOX_DIR, relative_path.lstrip('/'))

def log_operation(operation: str, file_path: str, success: bool, details: str = ""):
    """记录操作日志"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'file_path': file_path,
        'success': success,
        'details': details
    }
    
    log_file = os.path.join(SANDBOX_DIR, 'logs', 'file_operations.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

# =============================================================================
# 文件操作工具
# =============================================================================

class FileReadSchema(BaseModel):
    file_path: str = Field(description="要读取的文件路径（相对于工作区）")

@tool(args_schema=FileReadSchema)
def safe_read_file(file_path: str) -> str:
    """安全读取文件内容"""
    try:
        # 安全验证
        is_valid, error_msg = validate_path(file_path)
        if not is_valid:
            log_operation("read", file_path, False, error_msg)
            return f"❌ {error_msg}"
        
        is_valid_ext, ext_error = validate_file_extension(file_path)
        if not is_valid_ext:
            log_operation("read", file_path, False, ext_error)
            return f"❌ {ext_error}"
        
        safe_path = get_safe_file_path(file_path)
        
        # 检查文件是否存在
        if not os.path.exists(safe_path):
            log_operation("read", file_path, False, "文件不存在")
            return f"❌ 文件不存在: {file_path}"
        
        # 检查文件大小
        file_size_mb = os.path.getsize(safe_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            log_operation("read", file_path, False, f"文件过大: {file_size_mb:.2f}MB")
            return f"❌ 文件过大 ({file_size_mb:.2f}MB)，超过限制 ({MAX_FILE_SIZE_MB}MB)"
        
        # 读取文件
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        log_operation("read", file_path, True, f"成功读取 {len(content)} 字符")
        return f"✅ 文件内容:\n{content}"
        
    except Exception as e:
        error_msg = f"读取失败: {str(e)}"
        log_operation("read", file_path, False, error_msg)
        return f"❌ {error_msg}"

class FileWriteSchema(BaseModel):
    file_path: str = Field(description="要写入的文件路径（相对于工作区）")
    content: str = Field(description="要写入的文件内容")
    overwrite: bool = Field(default=False, description="是否覆盖已存在的文件")

@tool(args_schema=FileWriteSchema)
def safe_write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """安全写入文件"""
    try:
        # 安全验证
        is_valid, error_msg = validate_path(file_path)
        if not is_valid:
            log_operation("write", file_path, False, error_msg)
            return f"❌ {error_msg}"
        
        is_valid_ext, ext_error = validate_file_extension(file_path)
        if not is_valid_ext:
            log_operation("write", file_path, False, ext_error)
            return f"❌ {ext_error}"
        
        safe_path = get_safe_file_path(file_path)
        
        # 检查内容大小
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        if content_size_mb > MAX_FILE_SIZE_MB:
            log_operation("write", file_path, False, f"内容过大: {content_size_mb:.2f}MB")
            return f"❌ 内容过大 ({content_size_mb:.2f}MB)，超过限制 ({MAX_FILE_SIZE_MB}MB)"
        
        # 检查文件是否已存在
        if os.path.exists(safe_path) and not overwrite:
            log_operation("write", file_path, False, "文件已存在且未设置覆盖")
            return f"❌ 文件已存在: {file_path}。如需覆盖请设置 overwrite=True"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        # 写入文件
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        log_operation("write", file_path, True, f"成功写入 {len(content)} 字符")
        return f"✅ 文件已{'覆盖' if overwrite and os.path.exists(safe_path) else '创建'}: {file_path}"
        
    except Exception as e:
        error_msg = f"写入失败: {str(e)}"
        log_operation("write", file_path, False, error_msg)
        return f"❌ {error_msg}"

class FileDeleteSchema(BaseModel):
    file_path: str = Field(description="要删除的文件路径（相对于工作区）")
    confirm: bool = Field(description="确认删除操作")

@tool(args_schema=FileDeleteSchema)
def safe_delete_file(file_path: str, confirm: bool) -> str:
    """安全删除文件"""
    try:
        if not confirm:
            return "❌ 删除操作需要确认，请设置 confirm=True"
        
        # 安全验证
        is_valid, error_msg = validate_path(file_path)
        if not is_valid:
            log_operation("delete", file_path, False, error_msg)
            return f"❌ {error_msg}"
        
        safe_path = get_safe_file_path(file_path)
        
        # 检查文件是否存在
        if not os.path.exists(safe_path):
            log_operation("delete", file_path, False, "文件不存在")
            return f"❌ 文件不存在: {file_path}"
        
        # 额外安全检查 - 不能删除重要文件
        important_files = ['README.md', 'config.json', 'settings.ini']
        if os.path.basename(file_path) in important_files:
            log_operation("delete", file_path, False, "重要文件不允许删除")
            return f"❌ 重要文件不允许删除: {file_path}"
        
        # 删除文件
        os.remove(safe_path)
        
        log_operation("delete", file_path, True, "成功删除")
        return f"✅ 文件已删除: {file_path}"
        
    except Exception as e:
        error_msg = f"删除失败: {str(e)}"
        log_operation("delete", file_path, False, error_msg)
        return f"❌ {error_msg}"

class DirectoryListSchema(BaseModel):
    dir_path: str = Field(default=".", description="要列出的目录路径（相对于工作区）")
    show_details: bool = Field(default=False, description="显示详细信息（大小、修改时间等）")

@tool(args_schema=DirectoryListSchema)
def safe_list_directory(dir_path: str = ".", show_details: bool = False) -> str:
    """安全列出目录内容"""
    try:
        # 安全验证
        is_valid, error_msg = validate_path(dir_path)
        if not is_valid:
            log_operation("list", dir_path, False, error_msg)
            return f"❌ {error_msg}"
        
        safe_path = get_safe_file_path(dir_path)
        
        # 检查目录是否存在
        if not os.path.exists(safe_path):
            log_operation("list", dir_path, False, "目录不存在")
            return f"❌ 目录不存在: {dir_path}"
        
        if not os.path.isdir(safe_path):
            log_operation("list", dir_path, False, "不是目录")
            return f"❌ 不是目录: {dir_path}"
        
        # 列出目录内容
        entries = []
        entries.append(f"📁 目录内容: {dir_path}")
        entries.append("-" * 50)
        
        items = sorted(os.listdir(safe_path))
        for item in items:
            item_path = os.path.join(safe_path, item)
            
            if os.path.isdir(item_path):
                icon = "📁"
                type_info = "目录"
            else:
                icon = "📄"
                type_info = "文件"
            
            if show_details:
                # 获取详细信息
                stat_info = os.stat(item_path)
                size_mb = stat_info.st_size / (1024 * 1024)
                mod_time = datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M')
                entries.append(f"{icon} {item:<30} {type_info:<6} {size_mb:>8.2f}MB {mod_time}")
            else:
                entries.append(f"{icon} {item}")
        
        log_operation("list", dir_path, True, f"列出 {len(items)} 个项目")
        return "\n".join(entries)
        
    except Exception as e:
        error_msg = f"列出目录失败: {str(e)}"
        log_operation("list", dir_path, False, error_msg)
        return f"❌ {error_msg}"

class FileInfoSchema(BaseModel):
    file_path: str = Field(description="要查看信息的文件路径（相对于工作区）")

@tool(args_schema=FileInfoSchema)
def safe_file_info(file_path: str) -> str:
    """安全获取文件信息"""
    try:
        # 安全验证
        is_valid, error_msg = validate_path(file_path)
        if not is_valid:
            log_operation("info", file_path, False, error_msg)
            return f"❌ {error_msg}"
        
        safe_path = get_safe_file_path(file_path)
        
        # 检查文件是否存在
        if not os.path.exists(safe_path):
            log_operation("info", file_path, False, "文件不存在")
            return f"❌ 文件不存在: {file_path}"
        
        # 获取文件信息
        stat_info = os.stat(safe_path)
        
        info_lines = []
        info_lines.append(f"📋 文件信息: {file_path}")
        info_lines.append("-" * 40)
        info_lines.append(f"📁 类型: {'目录' if os.path.isdir(safe_path) else '文件'}")
        info_lines.append(f"📏 大小: {stat_info.st_size / (1024 * 1024):.2f} MB")
        info_lines.append(f"📅 创建时间: {datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
        info_lines.append(f"📝 修改时间: {datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        info_lines.append(f"🔒 权限: {oct(stat_info.st_mode)[-3:]}")
        
        if not os.path.isdir(safe_path):
            # 文件扩展名
            ext = pathlib.Path(file_path).suffix
            info_lines.append(f"📎 扩展名: {ext if ext else '无'}")
            
            # 如果是文本文件，显示行数
            if ext.lower() in {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml'}:
                try:
                    with open(safe_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    info_lines.append(f"📄 行数: {line_count}")
                except:
                    pass
        
        log_operation("info", file_path, True, "获取文件信息成功")
        return "\n".join(info_lines)
        
    except Exception as e:
        error_msg = f"获取文件信息失败: {str(e)}"
        log_operation("info", file_path, False, error_msg)
        return f"❌ {error_msg}"

class FileCopySchema(BaseModel):
    source_path: str = Field(description="源文件路径（相对于工作区）")
    dest_path: str = Field(description="目标文件路径（相对于工作区）")

@tool(args_schema=FileCopySchema)
def safe_copy_file(source_path: str, dest_path: str) -> str:
    """安全复制文件"""
    try:
        # 验证源文件路径
        is_valid, error_msg = validate_path(source_path)
        if not is_valid:
            log_operation("copy", f"{source_path} -> {dest_path}", False, f"源路径: {error_msg}")
            return f"❌ 源路径错误: {error_msg}"
        
        # 验证目标文件路径  
        is_valid, error_msg = validate_path(dest_path)
        if not is_valid:
            log_operation("copy", f"{source_path} -> {dest_path}", False, f"目标路径: {error_msg}")
            return f"❌ 目标路径错误: {error_msg}"
        
        safe_source = get_safe_file_path(source_path)
        safe_dest = get_safe_file_path(dest_path)
        
        # 检查源文件是否存在
        if not os.path.exists(safe_source):
            log_operation("copy", f"{source_path} -> {dest_path}", False, "源文件不存在")
            return f"❌ 源文件不存在: {source_path}"
        
        # 检查目标文件是否已存在
        if os.path.exists(safe_dest):
            log_operation("copy", f"{source_path} -> {dest_path}", False, "目标文件已存在")
            return f"❌ 目标文件已存在: {dest_path}"
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(safe_dest), exist_ok=True)
        
        # 复制文件
        shutil.copy2(safe_source, safe_dest)
        
        log_operation("copy", f"{source_path} -> {dest_path}", True, "复制成功")
        return f"✅ 文件已复制: {source_path} -> {dest_path}"
        
    except Exception as e:
        error_msg = f"复制失败: {str(e)}"
        log_operation("copy", f"{source_path} -> {dest_path}", False, error_msg)
        return f"❌ {error_msg}"

class DesktopWriteSchema(BaseModel):
    file_name: str = Field(description="要在桌面创建的文件名（包含扩展名）")
    content: str = Field(description="要写入的文件内容")
    overwrite: bool = Field(default=False, description="是否覆盖已存在的文件")

@tool(args_schema=DesktopWriteSchema)
def write_to_desktop(file_name: str, content: str, overwrite: bool = False) -> str:
    """直接写入文件到桌面"""
    try:
        # 验证文件名安全性
        if '..' in file_name or '/' in file_name or '\\' in file_name:
            return f"❌ 文件名不安全: {file_name}"
        
        # 验证文件扩展名
        is_valid_ext, ext_error = validate_file_extension(file_name)
        if not is_valid_ext:
            return f"❌ {ext_error}"
        
        desktop_path = os.path.join(DESKTOP_DIR, file_name)
        
        # 检查内容大小
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        if content_size_mb > MAX_FILE_SIZE_MB:
            return f"❌ 内容过大 ({content_size_mb:.2f}MB)，超过限制 ({MAX_FILE_SIZE_MB}MB)"
        
        # 检查文件是否已存在
        if os.path.exists(desktop_path) and not overwrite:
            return f"❌ 桌面文件已存在: {file_name}。如需覆盖请设置 overwrite=True"
        
        # 写入文件到桌面
        with open(desktop_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        log_operation("desktop_write", file_name, True, f"成功写入到桌面 {len(content)} 字符")
        return f"✅ 文件已{'覆盖' if overwrite and os.path.exists(desktop_path) else '创建'}到桌面: {file_name}"
        
    except Exception as e:
        error_msg = f"写入桌面失败: {str(e)}"
        log_operation("desktop_write", file_name, False, error_msg)
        return f"❌ {error_msg}"

@tool
def get_workspace_info() -> str:
    """获取工作区信息和使用指南"""
    ensure_sandbox()
    
    info_lines = []
    info_lines.append("🔒 安全文件管理工作区")
    info_lines.append("=" * 50)
    info_lines.append(f"📁 工作目录: {SANDBOX_DIR}")
    info_lines.append(f"📎 支持格式: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    info_lines.append(f"📏 文件大小限制: {MAX_FILE_SIZE_MB}MB")
    info_lines.append("")
    info_lines.append("🛠️ 可用操作:")
    info_lines.append("  • safe_read_file - 读取文件内容")
    info_lines.append("  • safe_write_file - 写入文件内容")
    info_lines.append("  • safe_delete_file - 删除文件")
    info_lines.append("  • safe_list_directory - 列出目录内容")
    info_lines.append("  • safe_file_info - 查看文件信息")
    info_lines.append("  • safe_copy_file - 复制文件")
    info_lines.append("")
    info_lines.append("🔐 安全特性:")
    info_lines.append("  • 沙盒环境限制")
    info_lines.append("  • 路径遍历防护")
    info_lines.append("  • 文件类型白名单")
    info_lines.append("  • 操作日志记录")
    info_lines.append("  • 文件大小限制")
    
    return "\n".join(info_lines)

# =============================================================================
# Agent创建
# =============================================================================

# 初始化工作区
ensure_sandbox()

SAFE_FILE_AGENT_PROMPT = """
你是一个专业的安全文件管理助手，负责在严格的安全环境下进行文件操作。

🔒 **安全原则**:
- 所有操作仅限在指定的安全工作区内
- 严格遵循文件类型和大小限制
- 记录所有操作日志以便审计
- 绝不执行可能危害系统安全的操作

🛠️ **核心功能**:
- 安全的文件读写操作
- 目录浏览和文件信息查询
- 文件复制和删除（需确认）
- 直接写入文件到桌面（特殊功能）
- 完整的操作日志记录

💡 **使用建议**:
- 使用相对路径（相对于工作区）
- 注意文件扩展名限制
- 删除操作需要明确确认
- 查看工作区信息了解限制

🚨 **安全限制**:
- 禁止访问系统目录和敏感文件
- 文件大小限制为10MB
- 仅支持安全的文件格式
- 所有路径都经过严格验证

当用户需要文件操作时，请严格按照安全规范执行，并提供清晰的操作反馈。
"""

# 创建工具列表
safe_file_tools = [
    get_workspace_info,
    safe_read_file,
    safe_write_file,
    safe_delete_file,
    safe_list_directory,
    safe_file_info,
    safe_copy_file,
    write_to_desktop
]

# 创建安全文件管理Agent
safe_file_agent = create_react_agent(
    model=model,
    tools=safe_file_tools,
    prompt=SAFE_FILE_AGENT_PROMPT,
    name="safe_file_agent"
)

if __name__ == "__main__":
    print("🔒 安全文件管理Agent 已启动")
    print(f"📁 工作目录: {SANDBOX_DIR}")
    print("安全特性:")
    print("- ✅ 沙盒环境保护")
    print("- ✅ 路径验证和过滤")
    print("- ✅ 文件类型白名单")
    print("- ✅ 操作日志记录")
    print("- ✅ 文件大小限制")
    print("- ✅ 重要文件保护")