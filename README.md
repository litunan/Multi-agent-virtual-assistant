# Multi-Agent 个人助手

多Agent协作系统，集成了多个专业Agent



### 系统概述

本系统采用智能Supervisor架构，集成了5个专业Agent：

- **🧠 ACP RAG Agent** - 抗癌肽领域专家
- **🐍 Python Agent** - 高级Python数据科学专家  
- **🗺️ AMAP Agent** - 全功能地理位置服务专家
- **🔒 Safe File Agent** - 安全文件管理专家
- **💻 Terminal Command Agent** - 安全终端命令执行专家
- 可以轻松接入其他Agent


### 系统架构

langgraph-supervisor架构

​	一个 supervisor 和多个 Agent，supervisor 负责管理多个 Agent，Agent 负责具体的工作，开始执行时，从 supervisor 开始，然后根据大模型的返回不断调各个 Agent，每个 Agent 执行完再次回到 supervisor，supervisor 再次调大模型，如此反复，直到得到结果。

### 框架细节

+ Supervisor

  📊 **系统架构**:
     • UserID - 用户身份标识（长期记忆和偏好）
     • SessionID - 会话标识（短期记忆和上下文）
     • ThreadID - 线程标识（默认为主线程 暂未完善） 

  🧠 **记忆层次**:
     • 用户级别: 长期知识、用户偏好、历史统计
     • 会话级别: 短期对话记忆、会话上下文
     • 线程级别: 暂未启用

  🔧 **管理功能**:
     • 用户仪表板（输入"仪表板"查看）
     • 会话生命周期管理
     • 数据自动清理和维护

  💡 **特殊命令**:
     • "仪表板" - 显示用户统计和偏好
     • "清理数据" - 清理旧数据 

  	1. 删除超过指定时间的会话 **针对会话表**
  	1. 清理旧的对话记忆 删除超过时间限制且重要性小于3的对话 **针对对话记忆表** 

     • "帮助" - 显示此帮助信息

+ **🗺️ AMAP Agent** - 全功能地理位置服务专家

  + 开发amap MCP服务器 

    ```python
    mcp = FastMCP("amap-maps")
    mcp.run(transport='streamable-http')
    ```

  + 接入agent

    ```python
    client = MultiServerMCPClient({
        "amap-maps-streamableHTTP": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable_http",
        },
    })
    ```

+ **🧠 ACP RAG Agent** - 医疗领域专家

  + 使用MinerU解析pdf文档

  + 使用text-embedding-v4进行嵌入

  + 递归切块 存入向量库(使用轻量化 FAISS 可替换)

  + Workflow：

    + LLM意图识别 决定是否调用检索工具

    + 检索：

      + 增强查询 提取关键词 意图分析 上下文加载 同义词获取

      + 根据意图进行不同的操作

        ```python
        if query_intent == 'definition':
            # 对于定义类查询，仅使用原始查询进行检索，简化处理
            full_query = search_query
        elif query_intent == 'comparison':
            # 对于比较类查询，扩展查询并加入更多上下文
            full_query = f"{search_query} {enhanced_terms} {synonyms}"
        elif ...
        ```

      + if 没有结果 则扩展搜索 使用关键词提取 并降低搜出文档相关性评分

      + 格式化 doc

    + 检索的结果如果置信度过低则问题重写 合理则生成答案

      评估过程使用LLM来完成 

    + 问题重写 LLM完成

    + 生成

    ```python
    # 构建增强版工作流
    enhanced_workflow = StateGraph(MessagesState)
    enhanced_workflow.add_node("generate_query_or_respond", enhanced_generate_query_or_respond)
    enhanced_workflow.add_node("retrieve", enhanced_retrieve_node)
    enhanced_workflow.add_node("rewrite_question", enhanced_rewrite_question)
    enhanced_workflow.add_node("generate_answer", enhanced_generate_answer)
    
    enhanced_workflow.add_edge(START, "generate_query_or_respond")
    enhanced_workflow.add_conditional_edges(
        "generate_query_or_respond", enhanced_tools_condition, {"tools": "retrieve", END: END}
    )
    enhanced_workflow.add_conditional_edges("retrieve", enhanced_grade_documents)
    enhanced_workflow.add_edge("generate_answer", END)
    enhanced_workflow.add_edge("rewrite_question", "generate_query_or_respond")
    
    # 编译增强版RAG Agent
    enhanced_rag_agent = enhanced_workflow.compile(name="enhanced_rag_agent")
    ```

+ **🔒 Safe File Agent** - 安全文件管理专家

  ```python
  # 创建安全文件管理Agent  新版本的langgraph的create_react_agent用langchain的create_agent替代
  safe_file_agent = create_react_agent(
      model=model,
      tools=safe_file_tools,
      prompt=SAFE_FILE_AGENT_PROMPT,
      name="safe_file_agent"
  )
  ```

### 消息传递展示

使用**langsmith**监控流程

