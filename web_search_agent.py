#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络检索Agent - 基于Tavily API的智能网络搜索专家
功能特性：
1. 实时网络搜索 - 获取最新信息
2. 智能内容提取 - 从指定URL提取内容  
3. 多维度搜索 - 支持新闻、金融、通用等不同主题
4. 搜索结果优化 - 专为AI和RAG应用优化
5. 灵活参数控制 - 搜索深度、时间范围、结果数量等
"""

import os
import json
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timedelta
from dotenv import load_dotenv 
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
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

# 获取Tavily API密钥
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_AVAILABLE = bool(TAVILY_API_KEY)
if not TAVILY_AVAILABLE:
    import logging
    logging.warning("未找到TAVILY_API_KEY环境变量，web_search_agent功能将受限")

# =============================================================================
# 基础网络搜索工具
# =============================================================================

class WebSearchSchema(BaseModel):
    query: str = Field(description="搜索查询关键词")
    max_results: int = Field(default=5, description="最大搜索结果数量（1-20）")
    topic: Literal["general", "news", "finance"] = Field(default="general", description="搜索主题类型")
    search_depth: Literal["basic", "advanced"] = Field(default="basic", description="搜索深度")
    include_answer: bool = Field(default=True, description="是否包含AI生成的答案摘要")
    include_images: bool = Field(default=False, description="是否包含相关图片")
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(default=None, description="时间范围限制")

@tool(args_schema=WebSearchSchema)
def web_search(
    query: str, 
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    search_depth: Literal["basic", "advanced"] = "basic",
    include_answer: bool = True,
    include_images: bool = False,
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
) -> str:
    """
    智能网络搜索 - 获取实时网络信息
    """
    if not TAVILY_AVAILABLE:
        return "❌ 网络搜索功能不可用：未配置 TAVILY_API_KEY。请在 .env 文件中配置 TAVILY_API_KEY。"
    
    try:
        # 创建Tavily搜索工具
        tavily_search = TavilySearch(
            api_key=TAVILY_API_KEY,
            max_results=max_results,
            topic=topic,
            include_answer=include_answer,
            search_depth=search_depth
        )
        
        # 构建搜索参数
        search_params = {"query": query}
        if include_images:
            search_params["include_images"] = include_images
        if time_range:
            search_params["time_range"] = time_range
        if search_depth != "basic":
            search_params["search_depth"] = search_depth
        
        # 执行搜索
        results = tavily_search.invoke(search_params)
        
        # 解析结果
        if isinstance(results, str):
            # 如果返回字符串，尝试解析为JSON
            try:
                results = json.loads(results)
            except:
                return f"✅ 搜索完成\n\n📋 搜索结果:\n{results}"
        
        # 格式化输出
        response = f"🔍 网络搜索完成 - 查询: \"{query}\"\n"
        response += f"📊 搜索参数: 主题={topic}, 深度={search_depth}, 结果数={max_results}\n\n"
        
        if isinstance(results, list):
            # 处理搜索结果列表
            for i, result in enumerate(results[:max_results], 1):
                if isinstance(result, dict):
                    title = result.get('title', '无标题')
                    url = result.get('url', '')
                    content = result.get('content', result.get('snippet', ''))
                    
                    response += f"📄 结果 {i}: {title}\n"
                    response += f"🔗 链接: {url}\n"
                    response += f"📝 摘要: {content[:200]}...\n\n"
                else:
                    response += f"📄 结果 {i}: {result}\n\n"
        elif isinstance(results, dict):
            # 处理单个结果字典
            if "answer" in results:
                response += f"🤖 AI答案摘要:\n{results['answer']}\n\n"
            
            if "results" in results:
                search_results = results["results"]
                for i, result in enumerate(search_results[:max_results], 1):
                    title = result.get('title', '无标题')
                    url = result.get('url', '')
                    content = result.get('content', result.get('snippet', ''))
                    
                    response += f"📄 结果 {i}: {title}\n"
                    response += f"🔗 链接: {url}\n"
                    response += f"📝 摘要: {content[:200]}...\n\n"
        else:
            response += f"📋 搜索结果:\n{str(results)}"
        
        return response
        
    except Exception as e:
        return f"❌ 网络搜索失败: {str(e)}"

# =============================================================================
# 新闻搜索工具
# =============================================================================

class NewsSearchSchema(BaseModel):
    query: str = Field(description="新闻搜索关键词")
    max_results: int = Field(default=5, description="最大新闻结果数量")
    time_range: Literal["day", "week", "month"] = Field(default="week", description="新闻时间范围")

@tool(args_schema=NewsSearchSchema)
def news_search(query: str, max_results: int = 5, time_range: Literal["day", "week", "month"] = "week") -> str:
    """
    专门的新闻搜索 - 获取最新新闻资讯
    """
    if not TAVILY_AVAILABLE:
        return "❌ 新闻搜索功能不可用：未配置 TAVILY_API_KEY。"
    
    try:
        tavily_search = TavilySearch(
            api_key=TAVILY_API_KEY,
            max_results=max_results,
            topic="news",
            include_answer=True,
            search_depth="advanced"
        )
        
        results = tavily_search.invoke({
            "query": query,
            "time_range": time_range
        })
        
        response = f"📰 新闻搜索完成 - 查询: \"{query}\"\n"
        response += f"⏰ 时间范围: 最近{time_range}\n\n"
        
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except:
                return response + results
        
        if isinstance(results, dict) and "results" in results:
            if "answer" in results:
                response += f"📝 新闻摘要:\n{results['answer']}\n\n"
            
            news_results = results["results"]
            for i, news in enumerate(news_results[:max_results], 1):
                title = news.get('title', '无标题')
                url = news.get('url', '')
                content = news.get('content', news.get('snippet', ''))
                published_date = news.get('published_date', '未知时间')
                
                response += f"📰 新闻 {i}: {title}\n"
                response += f"📅 发布时间: {published_date}\n"
                response += f"🔗 链接: {url}\n"
                response += f"📝 内容: {content[:250]}...\n\n"
        
        return response
        
    except Exception as e:
        return f"❌ 新闻搜索失败: {str(e)}"

# =============================================================================
# URL内容提取工具
# =============================================================================

class URLExtractSchema(BaseModel):
    urls: List[str] = Field(description="要提取内容的URL列表")
    max_content_length: int = Field(default=2000, description="每个URL提取内容的最大长度")

@tool(args_schema=URLExtractSchema)
def extract_url_content(urls: List[str], max_content_length: int = 2000) -> str:
    """
    从指定URL提取内容 - 智能网页内容抓取
    """
    if not TAVILY_AVAILABLE:
        return "❌ URL内容提取功能不可用：未配置 TAVILY_API_KEY。"
    
    try:
        if not urls:
            return "❌ URL列表不能为空"
        
        # 注意：Tavily的Extract功能需要专门的工具
        from langchain_tavily import TavilyExtract
        
        tavily_extract = TavilyExtract(api_key=TAVILY_API_KEY)
        
        results = tavily_extract.invoke({"urls": urls})
        
        response = f"📄 URL内容提取完成 - 处理 {len(urls)} 个链接\n\n"
        
        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    url = result.get('url', f'URL {i}')
                    title = result.get('title', '无标题')
                    content = result.get('content', '无内容')
                    
                    # 限制内容长度
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "...(内容已截断)"
                    
                    response += f"🔗 链接 {i}: {url}\n"
                    response += f"📋 标题: {title}\n"
                    response += f"📝 内容:\n{content}\n\n"
                    response += "-" * 50 + "\n\n"
        else:
            response += f"📄 提取结果:\n{str(results)}"
        
        return response
        
    except ImportError:
        # 如果没有TavilyExtract，使用基础搜索方式
        return "⚠️ URL内容提取功能需要更新langchain-tavily包版本"
    except Exception as e:
        return f"❌ URL内容提取失败: {str(e)}"

# =============================================================================
# 金融信息搜索工具
# =============================================================================

class FinanceSearchSchema(BaseModel):
    query: str = Field(description="金融相关搜索关键词")
    max_results: int = Field(default=5, description="最大搜索结果数量")

@tool(args_schema=FinanceSearchSchema)
def finance_search(query: str, max_results: int = 5) -> str:
    """
    金融信息搜索 - 获取股票、市场、经济相关信息
    """
    if not TAVILY_AVAILABLE:
        return "❌ 金融搜索功能不可用：未配置 TAVILY_API_KEY。"
    
    try:
        tavily_search = TavilySearch(
            api_key=TAVILY_API_KEY,
            max_results=max_results,
            topic="finance",
            include_answer=True,
            search_depth="advanced"
        )
        
        results = tavily_search.invoke({"query": query})
        
        response = f"💰 金融信息搜索完成 - 查询: \"{query}\"\n\n"
        
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except:
                return response + results
        
        if isinstance(results, dict) and "results" in results:
            if "answer" in results:
                response += f"📊 金融分析:\n{results['answer']}\n\n"
            
            finance_results = results["results"]
            for i, result in enumerate(finance_results[:max_results], 1):
                title = result.get('title', '无标题')
                url = result.get('url', '')
                content = result.get('content', result.get('snippet', ''))
                
                response += f"💹 信息 {i}: {title}\n"
                response += f"🔗 来源: {url}\n"
                response += f"📝 详情: {content[:200]}...\n\n"
        
        return response
        
    except Exception as e:
        return f"❌ 金融信息搜索失败: {str(e)}"

# =============================================================================
# 智能搜索建议工具
# =============================================================================

@tool
def get_search_suggestions(topic: str) -> str:
    """
    获取搜索建议和最佳实践
    """
    suggestions = {
        "新闻": [
            "添加具体时间关键词，如'2024年最新'",
            "包含地理位置，如'中国'、'全球'",
            "使用新闻相关词汇，如'报道'、'消息'、'事件'"
        ],
        "技术": [
            "包含版本信息，如'Python 3.12'",
            "添加'教程'、'文档'、'最佳实践'等关键词",
            "指定具体技术栈，如'React + TypeScript'"
        ],
        "金融": [
            "包含具体股票代码或公司名称",
            "添加时间范围，如'Q3 2024'",
            "使用金融专业术语，如'市值'、'PE比率'"
        ],
        "学术": [
            "添加'论文'、'研究'、'学术'等关键词",
            "包含具体领域，如'机器学习'、'生物医学'",
            "指定发表年份或期刊名称"
        ]
    }
    
    response = f"🎯 搜索建议 - 主题: {topic}\n\n"
    
    if topic in suggestions:
        response += f"💡 针对'{topic}'的搜索优化建议:\n"
        for suggestion in suggestions[topic]:
            response += f"• {suggestion}\n"
    else:
        response += "💡 通用搜索优化建议:\n"
        response += "• 使用具体而非泛泛的关键词\n"
        response += "• 包含时间限定词\n"
        response += "• 添加地理位置信息\n"
        response += "• 使用专业术语提高准确性\n"
    
    response += "\n🔧 可用搜索工具:\n"
    response += "• web_search - 通用网络搜索\n"
    response += "• news_search - 新闻资讯搜索\n"
    response += "• finance_search - 金融信息搜索\n"
    response += "• extract_url_content - URL内容提取\n"
    
    return response

# =============================================================================
# Agent配置
# =============================================================================

# 系统提示词
WEB_SEARCH_AGENT_PROMPT = """
你是一个专业的网络信息检索专家，基于Tavily API提供强大的实时网络搜索能力，禁止去成人网站检索。你具备以下核心能力：
🔍 **个人信息检索**:
- 实时获取最新网络信息(公众号，网页等来源)
🔍 **智能网络搜索**:
- 实时获取最新网络信息
- 支持通用、新闻、金融等不同主题搜索
- 灵活的搜索深度和结果数量控制
- AI优化的搜索结果，专为知识问答设计

📰 **专业新闻搜索**:
- 获取最新新闻资讯
- 支持时间范围筛选（最近一天/周/月）
- 提供新闻摘要和发布时间
- 深度搜索模式获取详细信息

💰 **金融信息搜索**:
- 专门的金融和市场信息检索
- 获取股票、经济、市场动态
- 提供专业的金融分析和见解

📄 **智能内容提取**:
- 从指定URL提取和分析内容
- 支持批量URL处理
- 智能内容摘要和结构化输出

**可用工具**:
1. **`web_search`** - 通用网络搜索（主要工具）
2. **`news_search`** - 专业新闻搜索
3. **`finance_search`** - 金融信息搜索
4. **`extract_url_content`** - URL内容提取
5. **`get_search_suggestions`** - 搜索建议和优化

**搜索特色**:
- ✅ 实时信息，比传统搜索引擎更新更快
- ✅ AI优化结果，直接提供答案摘要
- ✅ 多维度搜索，支持不同主题和深度
- ✅ 结构化输出，便于后续处理

**使用原则**:
- 根据用户查询类型选择最合适的搜索工具
- 优先使用 `web_search` 进行通用搜索
- 新闻相关查询使用 `news_search`
- 金融相关查询使用 `finance_search`
- 需要具体网页内容时使用 `extract_url_content`
- 提供清晰、结构化的搜索结果
- 在适当时候提供搜索优化建议

请根据用户的查询需求，选择最合适的工具来获取准确、及时的网络信息！
"""

# 创建Agent
web_search_agent = create_react_agent(
    model=model,
    tools=[
        web_search,
        news_search,
        finance_search,
        extract_url_content,
        get_search_suggestions
    ],
    name="web_search_agent"
)

if __name__ == "__main__":
    print("🚀 网络检索Agent已启动！")
    print("功能包括：")
    print("- 🔍 实时网络搜索（通用、新闻、金融）")
    print("- 📄 智能URL内容提取")
    print("- 🎯 搜索建议和优化")
    print("- 🤖 AI优化的搜索结果")
    print("- ⚡ 快速响应，专为AI应用设计")
    
    # 测试搜索功能
    print("\n🧪 测试网络搜索功能...")
    try:
        test_result = web_search.invoke({
            "query": "Python 3.12 新特性", 
            "max_results": 3
        })
        print("测试结果：", test_result[:300] + "..." if len(test_result) > 300 else test_result)
    except Exception as e:
        print(f"测试失败: {e}")