#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端命令执行Agent - 安全的系统命令执行和文件操作
功能特性：
1. 安全的终端命令执行
2. 文件系统操作（增删改查）
3. 命令白名单和黑名单机制
4. 执行结果实时反馈
5. 详细的操作日志记录
6. 多重安全防护
"""

import os
import sys
import json
import subprocess
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

# 工作目录沙盒 - 使用相对路径
SANDBOX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")

# 允许的命令白名单（安全命令）
ALLOWED_COMMANDS = {
    # 文件操作
    'ls', 'dir', 'pwd', 'cd', 'mkdir', 'rmdir', 'cp', 'mv', 'rm', 'touch',
    'cat', 'head', 'tail', 'grep', 'find', 'locate', 'which', 'file',
    # 文本处理
    'echo', 'printf', 'sort', 'uniq', 'wc', 'cut', 'awk', 'sed',
    # 网络工具
    'curl', 'wget', 'ping', 'nslookup', 'dig',
    # 开发工具
    'git', 'npm', 'pip', 'python', 'python3', 'node', 'java', 'javac',
    # 系统信息
    'ps', 'top', 'df', 'du', 'free', 'uname', 'whoami', 'id', 'date',
    # 压缩解压
    'tar', 'zip', 'unzip', 'gzip', 'gunzip',
    # 其他常用
    'chmod', 'chown', 'ln', 'diff', 'tree', 'history', 'stat', 'basename',
    'dirname', 'realpath', 'man', 'help', 'type', 'alias', 'env', 'printenv',
    # 进程和系统
    'kill', 'killall', 'jobs', 'bg', 'fg', 'nohup', 'screen', 'tmux',
    # 网络和连接
    'ssh', 'scp', 'rsync', 'telnet', 'ftp', 'sftp',
    # 编辑和查看
    'vi', 'vim', 'nano', 'emacs', 'less', 'more', 'hexdump', 'od'
}

# 危险命令黑名单
FORBIDDEN_COMMANDS = {
    # 系统管理
    'sudo', 'su', 'passwd', 'useradd', 'userdel', 'usermod', 'groupadd',
    'groupdel', 'mount', 'umount', 'fdisk', 'mkfs', 'fsck',
    # 网络配置
    'iptables', 'netstat', 'ss', 'route', 'ifconfig', 'ip',
    # 服务管理
    'systemctl', 'service', 'systemd', 'init', 'crontab',
    # 危险操作
    'dd', 'shred', 'format', 'fdisk', 'parted',
    # 系统关机重启
    'shutdown', 'reboot', 'halt', 'poweroff',
    # 包管理（系统级）
    'apt', 'yum', 'dnf', 'pacman', 'brew'
}

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.json', '.csv', '.log', '.py', '.js', '.html', '.css',
    '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.sh', '.bat'
}

# 命令执行超时时间（秒）
COMMAND_TIMEOUT = 30

# 最大输出长度
MAX_OUTPUT_LENGTH = 5000

def ensure_sandbox():
    """确保沙盒目录存在"""
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    
    # 创建子目录
    subdirs = ['documents', 'logs', 'data', 'temp', 'scripts']
    for subdir in subdirs:
        os.makedirs(os.path.join(SANDBOX_DIR, subdir), exist_ok=True)

def validate_command(command: str) -> tuple[bool, str]:
    """验证命令安全性"""
    if not command.strip():
        return False, "命令不能为空"
    
    # 获取命令的第一个词（实际命令）
    cmd_parts = command.strip().split()
    if not cmd_parts:
        return False, "无效的命令格式"
    
    base_cmd = cmd_parts[0].split('/')[-1]  # 处理路径形式的命令
    
    # 检查黑名单
    if base_cmd in FORBIDDEN_COMMANDS:
        return False, f"禁止执行危险命令: {base_cmd}"
    
    # 检查是否包含危险字符 - 放宽限制，允许管道等常用操作
    extremely_dangerous = ['&&', '||', ';', '`', '$(', 'rm -rf /', 'rm -rf *']
    for dangerous in extremely_dangerous:
        if dangerous in command:
            return False, f"命令包含极危险操作: {dangerous}"
    
    # 特殊检查：防止删除重要文件
    if 'rm' in command and any(pattern in command for pattern in ['-rf /', '-rf *', '-rf ~']):
        return False, "禁止执行可能删除系统文件的rm命令"
    
    # 检查白名单（更宽松，只对不在白名单的命令给出警告）
    if base_cmd not in ALLOWED_COMMANDS:
        # 检查是否是常见的安全命令变体
        safe_variants = ['ls', 'pwd', 'echo', 'cat', 'grep', 'find', 'head', 'tail']
        if any(safe in base_cmd for safe in safe_variants):
            return True, f"检测到命令变体: {base_cmd}"
        return True, f"警告: 命令 '{base_cmd}' 不在预定义安全列表中，请确认安全性"
    
    return True, ""

def log_operation(operation: str, command: str, success: bool, output: str = "", error: str = ""):
    """记录操作日志"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'command': command,
        'success': success,
        'output_length': len(output),
        'error': error[:500] if error else ""  # 限制错误信息长度
    }
    
    log_file = os.path.join(SANDBOX_DIR, 'logs', 'terminal_operations.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

# =============================================================================
# 终端命令执行工具
# =============================================================================

class CommandExecuteSchema(BaseModel):
    command: str = Field(description="要执行的终端命令")
    working_dir: str = Field(default="", description="执行命令的工作目录（相对于沙盒）")
    timeout: int = Field(default=30, description="命令执行超时时间（秒）")

@tool(args_schema=CommandExecuteSchema)
def execute_terminal_command(command: str, working_dir: str = "", timeout: int = 30) -> str:
    """
    安全执行终端命令 - 真实执行系统命令并返回结果
    """
    try:
        # 验证命令安全性
        is_valid, warning_msg = validate_command(command)
        if not is_valid:
            log_operation("execute_command", command, False, error=warning_msg)
            return f"❌ {warning_msg}"
        
        # 设置工作目录
        if working_dir:
            work_path = os.path.join(SANDBOX_DIR, working_dir.lstrip('/'))
            if not os.path.exists(work_path):
                os.makedirs(work_path, exist_ok=True)
        else:
            work_path = SANDBOX_DIR
        
        # 确保工作目录在沙盒内 - 但允许读取系统信息
        abs_work_path = os.path.abspath(work_path)
        abs_sandbox = os.path.abspath(SANDBOX_DIR)
        
        # 对于某些系统信息命令，允许在系统根目录执行
        system_info_commands = ['ps', 'top', 'df', 'free', 'uname', 'whoami', 'id', 'date', 'env', 'printenv']
        base_cmd = command.strip().split()[0].split('/')[-1]
        
        if base_cmd in system_info_commands:
            work_path = os.getcwd()  # 使用当前目录
        elif not abs_work_path.startswith(abs_sandbox):
            log_operation("execute_command", command, False, error="工作目录超出沙盒范围")
            return "❌ 工作目录超出安全范围"
        
        # 设置环境变量
        env = os.environ.copy()
        env['PWD'] = work_path
        
        # 执行命令
        print(f"🚀 执行命令: {command}")
        print(f"📁 工作目录: {work_path}")
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=work_path,
            capture_output=True,
            text=True,
            timeout=min(timeout, COMMAND_TIMEOUT),
            env=env
        )
        
        # 处理输出
        output = result.stdout.strip() if result.stdout else ""
        error = result.stderr.strip() if result.stderr else ""
        
        # 如果输出为空且命令成功，提供更友好的反馈
        if result.returncode == 0 and not output and not error:
            output = "(命令执行成功，无输出内容)"
        
        # 限制输出长度
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n... (输出被截断，共 " + str(len(result.stdout)) + " 字符)"
        
        if len(error) > MAX_OUTPUT_LENGTH:
            error = error[:MAX_OUTPUT_LENGTH] + "\n... (错误信息被截断)"
        
        # 记录日志
        log_operation("execute_command", command, result.returncode == 0, output, error)
        
        # 返回结果
        if result.returncode == 0:
            response = f"✅ 命令执行成功"
            if warning_msg:
                response += f" ({warning_msg})"
            if output:
                response += f"\n\n📋 输出结果:\n{output}"
            else:
                response += "\n\n📋 命令执行完成（无输出）"
            return response
        else:
            response = f"❌ 命令执行失败 (退出码: {result.returncode})"
            if error:
                response += f"\n\n❗ 错误信息:\n{error}"
            if output:
                response += f"\n\n📋 部分输出:\n{output}"
            return response
            
    except subprocess.TimeoutExpired:
        log_operation("execute_command", command, False, error="命令执行超时")
        return f"❌ 命令执行超时 ({timeout}秒) - 命令可能需要更长时间执行或陷入死循环"
    except FileNotFoundError as e:
        error_msg = f"命令未找到: {str(e)}"
        log_operation("execute_command", command, False, error=error_msg)
        return f"❌ {error_msg}\n💡 提示: 请检查命令是否拼写正确或已安装"
    except Exception as e:
        error_msg = f"执行失败: {str(e)}"
        log_operation("execute_command", command, False, error=error_msg)
        return f"❌ {error_msg}"

# =============================================================================
# 增强的交互式命令工具
# =============================================================================

class InteractiveCommandSchema(BaseModel):
    command: str = Field(description="要执行的交互式命令")
    input_data: str = Field(default="", description="要传递给命令的输入数据")
    working_dir: str = Field(default="", description="执行命令的工作目录")

@tool(args_schema=InteractiveCommandSchema)
def execute_interactive_command(command: str, input_data: str = "", working_dir: str = "") -> str:
    """
    执行可能需要输入的交互式命令
    """
    try:
        # 验证命令安全性
        is_valid, warning_msg = validate_command(command)
        if not is_valid:
            return f"❌ {warning_msg}"
        
        # 设置工作目录
        if working_dir:
            work_path = os.path.join(SANDBOX_DIR, working_dir.lstrip('/'))
            if not os.path.exists(work_path):
                os.makedirs(work_path, exist_ok=True)
        else:
            work_path = SANDBOX_DIR
        
        # 执行命令
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=work_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 如果有输入数据，传递给命令
        if input_data:
            stdout, stderr = process.communicate(input=input_data, timeout=COMMAND_TIMEOUT)
        else:
            stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT)
        
        # 处理输出
        output = stdout.strip() if stdout else ""
        error = stderr.strip() if stderr else ""
        
        # 记录日志
        log_operation("execute_interactive", command, process.returncode == 0, output, error)
        
        if process.returncode == 0:
            response = f"✅ 交互式命令执行成功"
            if warning_msg:
                response += f" ({warning_msg})"
            if output:
                response += f"\n\n📋 输出结果:\n{output}"
            return response
        else:
            return f"❌ 交互式命令执行失败 (退出码: {process.returncode})\n❗ 错误: {error}"
            
    except subprocess.TimeoutExpired:
        return f"❌ 交互式命令执行超时"
    except Exception as e:
        return f"❌ 交互式命令执行失败: {str(e)}"

# =============================================================================
# 批量命令执行工具
# =============================================================================

class BatchCommandSchema(BaseModel):
    commands: List[str] = Field(description="要批量执行的命令列表")
    stop_on_error: bool = Field(default=True, description="遇到错误时是否停止执行")
    working_dir: str = Field(default="", description="执行命令的工作目录")

@tool(args_schema=BatchCommandSchema)
def execute_batch_commands(commands: List[str], stop_on_error: bool = True, working_dir: str = "") -> str:
    """
    批量执行多个终端命令
    """
    if not commands:
        return "❌ 命令列表不能为空"
    
    results = []
    executed = 0
    failed = 0
    
    for i, command in enumerate(commands):
        results.append(f"\n{'='*50}")
        results.append(f"📋 执行命令 {i+1}/{len(commands)}: {command}")
        results.append(f"{'='*50}")
        
        # 执行单个命令
        result = execute_terminal_command(command, working_dir)
        results.append(result)
        
        executed += 1
        
        # 检查是否失败
        if result.startswith("❌"):
            failed += 1
            if stop_on_error:
                results.append(f"\n⚠️  遇到错误，停止执行剩余 {len(commands) - i - 1} 个命令")
                break
    
    # 添加执行摘要
    results.insert(0, f"🎯 批量执行摘要: 执行 {executed}/{len(commands)} 个命令，{failed} 个失败")
    
    return "\n".join(results)

# =============================================================================
# 文件操作工具
# =============================================================================

class FileOperationSchema(BaseModel):
    operation: str = Field(description="操作类型: read, write, delete, list, copy, move")
    file_path: str = Field(description="文件路径（相对于沙盒）")
    content: str = Field(default="", description="写入的内容（仅用于write操作）")
    target_path: str = Field(default="", description="目标路径（仅用于copy/move操作）")

@tool(args_schema=FileOperationSchema)
def file_operation(operation: str, file_path: str, content: str = "", target_path: str = "") -> str:
    """
    安全的文件操作工具
    """
    try:
        # 构建安全路径
        safe_path = os.path.join(SANDBOX_DIR, file_path.lstrip('/'))
        abs_safe_path = os.path.abspath(safe_path)
        abs_sandbox = os.path.abspath(SANDBOX_DIR)
        
        # 验证路径安全性
        if not abs_safe_path.startswith(abs_sandbox):
            return "❌ 文件路径超出安全范围"
        
        if operation == "read":
            if not os.path.exists(safe_path):
                return f"❌ 文件不存在: {file_path}"
            
            if os.path.isdir(safe_path):
                return f"❌ 指定路径是目录，不是文件: {file_path}"
            
            with open(safe_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            log_operation("file_read", file_path, True)
            return f"✅ 文件内容:\n{file_content}"
        
        elif operation == "write":
            # 确保目录存在
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            log_operation("file_write", file_path, True)
            return f"✅ 文件写入成功: {file_path}"
        
        elif operation == "delete":
            if not os.path.exists(safe_path):
                return f"❌ 文件不存在: {file_path}"
            
            if os.path.isdir(safe_path):
                shutil.rmtree(safe_path)
                log_operation("dir_delete", file_path, True)
                return f"✅ 目录删除成功: {file_path}"
            else:
                os.remove(safe_path)
                log_operation("file_delete", file_path, True)
                return f"✅ 文件删除成功: {file_path}"
        
        elif operation == "list":
            if not os.path.exists(safe_path):
                return f"❌ 路径不存在: {file_path}"
            
            if os.path.isfile(safe_path):
                return f"✅ {file_path} 是一个文件"
            
            items = []
            for item in os.listdir(safe_path):
                item_path = os.path.join(safe_path, item)
                if os.path.isdir(item_path):
                    items.append(f"📁 {item}/")
                else:
                    items.append(f"📄 {item}")
            
            log_operation("dir_list", file_path, True)
            return f"✅ 目录内容:\n" + "\n".join(items)
        
        elif operation == "copy":
            if not target_path:
                return "❌ 复制操作需要指定目标路径"
            
            safe_target = os.path.join(SANDBOX_DIR, target_path.lstrip('/'))
            abs_safe_target = os.path.abspath(safe_target)
            
            if not abs_safe_target.startswith(abs_sandbox):
                return "❌ 目标路径超出安全范围"
            
            if not os.path.exists(safe_path):
                return f"❌ 源文件不存在: {file_path}"
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(safe_target), exist_ok=True)
            
            if os.path.isdir(safe_path):
                shutil.copytree(safe_path, safe_target)
            else:
                shutil.copy2(safe_path, safe_target)
            
            log_operation("file_copy", f"{file_path} -> {target_path}", True)
            return f"✅ 复制成功: {file_path} -> {target_path}"
        
        elif operation == "move":
            if not target_path:
                return "❌ 移动操作需要指定目标路径"
            
            safe_target = os.path.join(SANDBOX_DIR, target_path.lstrip('/'))
            abs_safe_target = os.path.abspath(safe_target)
            
            if not abs_safe_target.startswith(abs_sandbox):
                return "❌ 目标路径超出安全范围"
            
            if not os.path.exists(safe_path):
                return f"❌ 源文件不存在: {file_path}"
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(safe_target), exist_ok=True)
            
            shutil.move(safe_path, safe_target)
            
            log_operation("file_move", f"{file_path} -> {target_path}", True)
            return f"✅ 移动成功: {file_path} -> {target_path}"
        
        else:
            return f"❌ 不支持的操作类型: {operation}"
            
    except Exception as e:
        error_msg = f"文件操作失败: {str(e)}"
        log_operation(f"file_{operation}", file_path, False, error=error_msg)
        return f"❌ {error_msg}"

# =============================================================================
# 系统信息工具
# =============================================================================

@tool
def get_system_info() -> str:
    """
    获取系统信息和当前状态
    """
    try:
        info = []
        info.append("🖥️ 系统信息:")
        info.append(f"操作系统: {os.name}")
        info.append(f"当前工作目录: {os.getcwd()}")
        info.append(f"沙盒目录: {SANDBOX_DIR}")
        info.append(f"Python版本: {sys.version}")
        
        # 沙盒目录状态
        if os.path.exists(SANDBOX_DIR):
            items = os.listdir(SANDBOX_DIR)
            info.append(f"\n📁 沙盒目录内容 ({len(items)} 项):")
            for item in items[:10]:  # 只显示前10项
                item_path = os.path.join(SANDBOX_DIR, item)
                if os.path.isdir(item_path):
                    info.append(f"  📁 {item}/")
                else:
                    info.append(f"  📄 {item}")
            if len(items) > 10:
                info.append(f"  ... 还有 {len(items) - 10} 项")
        
        return "\n".join(info)
        
    except Exception as e:
        return f"❌ 获取系统信息失败: {str(e)}"

# =============================================================================
# Agent配置
# =============================================================================

# 确保沙盒环境
ensure_sandbox()

# 系统提示词
TERMINAL_AGENT_PROMPT = """
你是一个专业的终端命令执行和文件管理专家。你具备以下核心能力：

🖥️ **终端命令执行**:
- 安全执行各种终端命令，真实调用系统命令
- 支持文件操作、系统查询、开发工具等
- 自动验证命令安全性
- 提供详细的执行结果反馈
- 支持交互式命令和批量执行

📁 **文件系统操作**:
- 文件的增删改查操作
- 目录管理和文件组织
- 安全的文件复制和移动
- 文件内容读取和编辑

🛡️ **安全特性**:
- 沙盒环境保护（文件操作限制在工作区）
- 命令白名单和黑名单机制
- 路径遍历攻击防护
- 执行超时保护（30秒）
- 详细的操作日志记录

**可用工具**:
1. **`execute_terminal_command`** - 执行单个终端命令（主要工具）
2. **`execute_interactive_command`** - 执行需要输入的交互式命令
3. **`execute_batch_commands`** - 批量执行多个命令
4. **`file_operation`** - 文件操作（读写删除复制移动等）
5. **`get_system_info`** - 获取系统和沙盒信息

**命令执行特点**:
- ✅ 真实执行系统命令，返回实际输出
- ✅ 支持ls、pwd、cat、grep、ps、df等常用命令
- ✅ 支持python、git、npm等开发工具
- ✅ 允许管道操作 (|) 和重定向 (>, <)
- ❌ 禁止sudo、rm -rf /等危险操作

**使用原则**:
- 优先使用 `execute_terminal_command` 执行用户请求的命令
- 始终确保操作的安全性
- 提供清晰的执行结果反馈
- 遇到危险操作时主动警告并拒绝执行
- 保持操作的可追溯性

请根据用户需求，选择合适的工具来完成任务！
"""

# 创建Agent
terminal_command_agent = create_react_agent(
    model=model,
    tools=[
        execute_terminal_command,
        execute_interactive_command,
        execute_batch_commands,
        file_operation,
        get_system_info
    ],
    name="terminal_command_agent"
)

if __name__ == "__main__":
    print("🚀 增强版终端命令Agent已启动！")
    print("功能包括：")
    print("- ✅ 真实的终端命令执行（支持ls、ps、df等）")
    print("- 🔄 交互式命令支持")
    print("- 📦 批量命令执行")
    print("- 📁 完整的文件系统操作")
    print("- 📊 系统信息查询")
    print("- 🛡️ 多重安全防护机制")
    print("- 📝 详细的操作日志记录")
    
    # 演示命令执行
    print("\n🧪 测试命令执行功能...")
    test_result = execute_terminal_command("ls -la")
    print("测试结果：", test_result[:200] + "..." if len(test_result) > 200 else test_result)