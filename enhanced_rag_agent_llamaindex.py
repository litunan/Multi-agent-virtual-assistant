"""
增强版 RAG Agent - LlamaIndex 实现
使用 LlamaIndex FAISS 向量数据库进行知识检索
作者: AI Assistant
日期: 2025-12-22
"""
from __future__ import annotations
import os
import json
import asyncio
from typing import Literal, List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import re
from datetime import datetime
from collections import defaultdict
import math
from langchain_openai import ChatOpenAI

# LlamaIndex imports
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from config.load_key import load_key

# 初始化模型
model = ChatOpenAI(
    # 若没有配置环境变量,请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=load_key("aliyun-bailian"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 阿里云百炼模型
)



grader_model = ChatOpenAI(
    # 若没有配置环境变量,请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=load_key("aliyun-bailian"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 阿里云百炼模型
)




class EnhancedTextRetrieverLlamaIndex(BaseTool):
    """增强版 LlamaIndex 向量数据库检索工具 - 抗癌肽专用"""
    name: str = "retrieve_anticancer_peptides"
    description: str = "Search and return relevant information from the Anticancer Peptides knowledge base using LlamaIndex FAISS vector database."
    conversation_history: List[str] = []
    context_cache: Dict[str, Any] = {}
    index: Any = None
    retriever: Any = None

    def __init__(self):
        """初始化检索工具"""
        super().__init__()
        self.conversation_history = []
        self.context_cache = {}
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """初始化 LlamaIndex FAISS 向量数据库"""
        try:
            # 初始化 embeddings - 使用本地 HuggingFace 模型
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu",
                normalize=True,
            )
            Settings.embed_model = embed_model

            # 加载 LlamaIndex FAISS 向量数据库
            output_dir = "mcp_course_materials_db_llamaindex"
            vector_store = FaissVectorStore.from_persist_dir(output_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=output_dir,
            )
            
            self.index = load_index_from_storage(storage_context, embed_model=embed_model)
            self.retriever = self.index.as_retriever(similarity_top_k=5)
            print("LlamaIndex 向量数据库加载成功")
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}")
            self.index = None
            self.retriever = None

    def _run(self, query: str, run_manager: CallbackManagerForToolRun = None) -> str:
        """检索相关文档内容"""
        try:
            # 保存对话历史
            self.conversation_history.append(query)

            if self.retriever is None:
                return "向量数据库未初始化,请检查配置。"

            # 增强查询处理
            enhanced_query = self._enhance_query(query)

            # 执行向量搜索
            results = self._vector_search(enhanced_query, top_k=5)

            if not results:
                # 尝试扩展搜索
                expanded_results = self._expanded_search(query)
                if expanded_results:
                    return self._format_expanded_results(expanded_results, query)
                return f"未找到与'{query}'直接相关的信息。您可以尝试使用更具体的关键词，或询问以下相关问题：\n" + self._suggest_questions()

            # 格式化结果
            formatted_result = self._format_enhanced_results(results, query)

            # 缓存上下文
            self.context_cache[query] = results

            return formatted_result

        except Exception as e:
            return f"检索时发生错误：{str(e)}"

    def _enhance_query(self, query: str) -> Dict[str, Any]:
        """增强查询处理"""
        enhanced = {
            'original': query,
            'terms': self._extract_key_terms(query),
            'intent': self._analyze_intent(query),
            'context': self._get_conversation_context(),
            'synonyms': self._get_synonyms(query)
        }
        return enhanced

    def _extract_key_terms(self, query: str) -> List[str]:
        """提取关键术语 - 抗癌肽领域"""
        technical_terms = {
            '抗癌肽': ['anticancer peptides', 'ACPs', '抗肿瘤肽', '抗癌多肽'],
            '肽': ['peptide', '多肽', '短肽', '蛋白质片段'],
            '癌症': ['cancer', '肿瘤', '恶性肿瘤', '癌变'],
            '细胞': ['cell', '细胞系', '癌细胞', '肿瘤细胞'],
            '机制': ['mechanism', '作用机制', '机理', '分子机制'],
            '毒性': ['toxicity', '细胞毒性', '毒副作用', '安全性'],
            '选择性': ['selectivity', '特异性', '靶向性', '选择性杀伤'],
            '结构': ['structure', '构效关系', '二级结构', '空间结构'],
            '活性': ['activity', '生物活性', '抗癌活性', '抑制活性'],
            '治疗': ['therapy', '治疗', '药物治疗', '靶向治疗'],
            '预测': ['prediction', '识别', '分类', '机器学习'],
            '数据库': ['database', '数据集', '资源库', '信息库'],
            '设计': ['design', '理性设计', '优化设计', '肽设计'],
            '穿透': ['penetration', '细胞穿透', '膜穿透', '细胞内化'],
            '耐药性': ['resistance', '耐药', '药物抵抗', '治疗抵抗'],
            '生物信息学': ['bioinformatics', '计算生物学', '生物计算'],
            '分子对接': ['docking', '分子模拟', '计算机辅助设计'],
            '临床试验': ['clinical trial', '临床研究', '人体试验']
        }

        terms = []
        query_lower = query.lower()

        for key, synonyms in technical_terms.items():
            if key in query or any(syn in query_lower for syn in synonyms):
                terms.extend([key] + synonyms)

        # 添加原始查询词
        terms.extend(query.split())

        return list(set(terms))

    def _analyze_intent(self, query: str) -> str:
        """分析查询意图"""
        intent_patterns = {
            'definition': ['是什么', '定义', '含义', '解释', '概念'],
            'method': ['如何', '怎么', '方法', '步骤', '流程'],
            'comparison': ['比较', '区别', '差异', '对比', '异同'],
            'analysis': ['分析', '评估', '研究', '探讨', '调查'],
            'example': ['例子', '示例', '案例', '举例', '实例'],
            'reason': ['为什么', '原因', '因素', '影响', '导致'],
            'process': ['过程', '流程', '步骤', '阶段', '环节'],
            'mechanism': ['机制', '原理', '作用方式', '分子机制']
        }

        query_lower = query.lower()
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        return 'general'

    def _get_conversation_context(self) -> List[str]:
        """获取对话上下文"""
        return self.conversation_history[-3:] if len(self.conversation_history) > 1 else []

    def _get_synonyms(self, query: str) -> List[str]:
        """获取同义词"""
        synonym_dict = {
            '抗癌肽': ['抗肿瘤肽', '抗癌多肽', 'ACPs'],
            '肽': ['多肽', '蛋白质片段', '短肽'],
            '癌症': ['肿瘤', '恶性肿瘤', '癌'],
            '机制': ['机理', '作用机制', '分子机制'],
            '毒性': ['细胞毒性', '毒副作用'],
            '选择性': ['特异性', '靶向性'],
            '活性': ['生物活性', '抑制活性'],
            '预测': ['识别', '分类', '鉴定']
        }

        synonyms = []
        for word, syns in synonym_dict.items():
            if word in query:
                synonyms.extend(syns)

        return synonyms

    def _vector_search(self, enhanced_query: Dict, top_k: int = 5) -> List[Dict]:
        """使用 LlamaIndex FAISS 进行向量搜索"""
        if self.retriever is None:
            return []

        try:
            # 使用增强的查询进行搜索
            search_query = enhanced_query['original']

            # 获取关键术语和同义词，组合成一个增强查询
            enhanced_terms = ' '.join(enhanced_query['terms'])
            synonyms = ' '.join(enhanced_query['synonyms'])

            # 获取意图
            query_intent = enhanced_query['intent']

            # 基于意图分析调整查询策略
            if query_intent == 'definition':
                full_query = search_query
            elif query_intent == 'comparison':
                full_query = f"{search_query} {enhanced_terms} {synonyms}"
            elif query_intent == 'analysis':
                full_query = f"{search_query} {enhanced_terms} {synonyms}"
            else:
                full_query = f"{search_query} {enhanced_terms} {synonyms}"
            
            # 执行 LlamaIndex 检索
            nodes = self.retriever.retrieve(full_query)

            results = []
            for i, node in enumerate(nodes[:top_k]):
                # 获取节点分数
                relevance_score = float(node.score) if hasattr(node, 'score') else (1.0 - i * 0.1)

                results.append({
                    'content': node.get_content(),
                    'metadata': node.metadata,
                    'relevance': min(relevance_score, 1.0),
                    'score': relevance_score,
                    'title': node.metadata.get('file_name', '未知来源'),
                    'chunk_id': i
                })

            return results

        except Exception as e:
            print(f"向量搜索错误: {str(e)}")
            return []

    def _expanded_search(self, query: str) -> List[Dict]:
        """扩展搜索策略"""
        if self.retriever is None:
            return []

        try:
            # 尝试使用关键术语进行搜索
            enhanced_query = self._enhance_query(query)
            terms = enhanced_query['terms']

            expanded_results = []
            for term in terms[:3]:  # 只使用前3个关键术语
                nodes = self.retriever.retrieve(term)
                for node in nodes[:2]:
                    expanded_results.append({
                        'content': node.get_content(),
                        'metadata': node.metadata,
                        'relevance': 0.5,
                        'score': 0.5,
                        'title': node.metadata.get('file_name', '未知来源'),
                        'search_term': term
                    })

            # 去重
            unique_results = []
            seen_content = set()
            for result in expanded_results:
                content_hash = hash(result['content'][:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)

            return unique_results[:3]

        except Exception as e:
            print(f"扩展搜索错误: {str(e)}")
            return []

    def _format_enhanced_results(self, results: List[Dict], query: str) -> str:
        """格式化增强搜索结果"""
        if not results:
            return "未找到相关信息。"

        formatted_results = []
        formatted_results.append(f"针对查询「{query}」找到以下相关信息：\n")

        for i, result in enumerate(results, 1):
            relevance_score = result.get('relevance', 0)
            confidence = "高" if relevance_score > 0.7 else "中" if relevance_score > 0.4 else "低"

            formatted_results.append(f"📌 【相关内容 {i}】（相关度：{confidence}）")

            # 显示来源信息
            source = result.get('metadata', {}).get('file_name', '未知来源')
            formatted_results.append(f"来源: {source}")

            # LlamaIndex 的节点通常更长，截取更多内容
            content = result['content']
            preview = content[:600] if len(content) > 600 else content
            formatted_results.append(f"{preview}...")

            if i < len(results):
                formatted_results.append("")

        # 添加相关问题推荐
        formatted_results.append(f"\n您可能还想了解：")
        suggestions = self._generate_suggestions(query, results)
        for suggestion in suggestions:
            formatted_results.append(f"   • {suggestion}")

        return "\n".join(formatted_results)

    def _format_expanded_results(self, results: List[Dict], query: str) -> str:
        """格式化扩展搜索结果"""
        formatted_results = []
        formatted_results.append(f"未找到「{query}」的直接匹配，但找到以下相关信息：\n")

        for i, result in enumerate(results, 1):
            search_term = result.get('search_term', '相关术语')

            formatted_results.append(f"【扩展结果 {i}】（基于: {search_term}）")

            source = result.get('metadata', {}).get('file_name', '未知来源')
            formatted_results.append(f"📄 来源: {source}")
            formatted_results.append(f"📝 {result['content'][:400]}...")
            formatted_results.append("")

        return "\n".join(formatted_results)

    def _suggest_questions(self) -> str:
        """生成建议问题"""
        suggestions = [
            "抗癌肽的主要作用机制是什么？",
            "如何设计和优化抗癌肽的结构？",
            "抗癌肽的细胞选择性是如何实现的？",
            "抗癌肽临床试验的最新进展有哪些？",
            "如何预测和评估抗癌肽的活性？"
        ]
        return "\n".join([f"   ❓ {q}" for q in suggestions])

    def _generate_suggestions(self, query: str, results: List[Dict]) -> List[str]:
        """基于查询和结果生成相关问题建议"""
        suggestions = []

        # 基于查询意图生成建议
        if "抗癌肽" in query or "肽" in query:
            suggestions.extend([
                "抗癌肽与传统化疗药物相比有什么优势？",
                "抗癌肽的给药方式和剂型有哪些？"
            ])

        if "机制" in query or "作用" in query:
            suggestions.extend([
                "抗癌肽如何诱导肿瘤细胞凋亡？",
                "抗癌肽的膜破坏机制具体是怎样的？"
            ])

        if "设计" in query or "预测" in query:
            suggestions.extend([
                "基于机器学习的抗癌肽设计方法有哪些？",
                "如何提高抗癌肽的稳定性和半衰期？"
            ])

        if "毒性" in query or "安全" in query:
            suggestions.extend([
                "如何评估抗癌肽对正常细胞的毒性？",
                "抗癌肽的免疫原性如何控制？"
            ])

        # 基于结果内容生成建议
        for result in results[:2]:
            content = result['content'].lower()
            if "结构" in content and "结构相关问题" not in [s for s in suggestions]:
                suggestions.append("抗癌肽的构效关系有哪些规律？")
            if "数据库" in content and "数据库相关问题" not in [s for s in suggestions]:
                suggestions.append("有哪些常用的抗癌肽数据库资源？")

        return suggestions[:3]


# 创建增强版检索工具实例（LlamaIndex版本）
enhanced_retriever_tool = EnhancedTextRetrieverLlamaIndex()

# 增强版Prompt - 抗癌肽专家
ENHANCED_SYSTEM_INSTRUCTION = (
    "你是一个专业的抗癌肽研究专家助手，专精于抗癌肽的设计、机制研究和临床应用。你具备以下能力：\n\n"
    "🔬 **核心专长**:\n"
    "- 深度理解抗癌肽的作用机制和分子生物学\n"
    "- 熟悉抗癌肽的理性设计和优化策略\n"
    "- 精通生物信息学和计算生物学方法\n"
    "- 擅长抗癌肽的活性评估和安全性分析\n\n"
    "💡 **回答特色**:\n"
    "- 提供结构化、层次分明的专业解答\n"
    "- 结合分子机制和临床前景\n"
    "- 主动提供相关问题建议\n"
    "- 解释专业术语和生物学术语\n\n"
    "⚙️ **工作方式**:\n"
    "- 仔细分析用户问题的科学内涵\n"
    "- 从向量数据库中检索最相关的信息\n"
    "- 提供全面而精准的专业解答\n"
    "- 主动推荐相关的深入研究方向\n\n"
    "如果问题不在抗癌肽研究范围内，请礼貌说明并引导到相关主题。\n"
    "当需要更多信息时，请调用检索工具 `retrieve_anticancer_peptides`。"
)

ENHANCED_GRADE_PROMPT = (
    "你是一个专业的信息相关性评估专家。请评估检索到的文档内容与用户问题的相关性。\n\n"
    "评估标准：\n"
    "- 内容是否直接回答了用户的问题\n"
    "- 信息的准确性和科学性\n"
    "- 是否包含用户需要的关键信息\n\n"
    "检索到的文档：\n{context}\n\n"
    "用户问题：{question}\n\n"
    "如果内容高度相关且有科学价值，返回 'yes'；否则返回 'no'。"
)

ENHANCED_REWRITE_PROMPT = (
    "你是一个查询优化专家。请将用户的问题重写得更加精确和易于检索。\n\n"
    "优化原则：\n"
    "- 使用专业的生物医学和分子生物学术语\n"
    "- 明确查询的科学意图\n"
    "- 增加相关的同义词和关键词\n"
    "- 保持问题的原始含义\n\n"
    "原始问题：\n{question}\n\n"
    "请提供优化后的查询版本："
)

ENHANCED_ANSWER_PROMPT = (
    "你是一位资深的抗癌肽研究专家。请基于检索到的上下文信息，为用户提供专业、详细的解答。\n\n"
    "回答要求：\n"
    "📋 **结构化回答**：使用清晰的标题和分点说明\n"
    "🔬 **科学深度**：包含分子机制、实验证据和临床意义\n"
    "💡 **实用建议**：提供研究思路和实验设计建议\n"
    "📊 **数据支持**：引用相关研究数据和文献支持\n"
    "🚀 **前沿展望**：关联最新研究进展和未来方向\n\n"
    "如果上下文信息不足，请诚实说明并建议用户提供更多信息或调整问题。\n\n"
    "用户问题：\n{question}\n\n"
    "检索上下文：\n{context}\n\n"
    "请提供专业的科学解答："
)


# 节点函数
async def enhanced_generate_query_or_respond(state: MessagesState):
    """LLM决定直接回答还是调用检索工具"""
    response = await model.bind_tools([enhanced_retriever_tool]).ainvoke(
        [
            {"role": "system", "content": ENHANCED_SYSTEM_INSTRUCTION},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


class EnhancedGradeDoc(BaseModel):
    """文档相关性评分模型"""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the relevance assessment")


async def enhanced_grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """增强版文档相关性评估"""
    question = state["messages"][0].content
    ctx = state["messages"][-1].content
    prompt = ENHANCED_GRADE_PROMPT.format(question=question, context=ctx)

    result = await grader_model.with_structured_output(EnhancedGradeDoc).ainvoke([
        {"role": "user", "content": prompt}
    ])

    # 基于置信度和推理进行更智能的判断
    if result.binary_score.lower().startswith("y") and result.confidence > 0.6:
        return "generate_answer"
    else:
        return "rewrite_question"


async def enhanced_rewrite_question(state: MessagesState):
    """增强版问题重写"""
    question = state["messages"][0].content
    prompt = ENHANCED_REWRITE_PROMPT.format(question=question)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": resp.content}]}


async def enhanced_generate_answer(state: MessagesState):
    """增强版答案生成"""
    question = state["messages"][0].content
    ctx = state["messages"][-1].content
    prompt = ENHANCED_ANSWER_PROMPT.format(question=question, context=ctx)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}


# 工具节点
async def enhanced_retrieve_node(state: MessagesState):
    """增强版检索节点"""
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        query = tool_call['args'].get('query', '')

        # 调用增强版检索工具
        result = enhanced_retriever_tool._run(query)

        return {"messages": [{"role": "tool", "content": result, "tool_call_id": tool_call['id']}]}
    else:
        return {"messages": [{"role": "tool", "content": "未找到有效的检索查询"}]}


def enhanced_tools_condition(state: MessagesState):
    """工具条件判断"""
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return END


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
enhanced_rag_agent = enhanced_workflow.compile(name="enhanced_rag_agent_llamaindex")


# 测试函数
async def test_enhanced_rag_agent():
    """测试增强版RAG Agent (LlamaIndex版本)"""
    print("=" * 80)
    print("测试增强版RAG Agent (LlamaIndex版本)")
    print("=" * 80)

    test_queries = [
        "抗癌肽是什么",
        # "什么是抗癌肽？它们的主要作用机制是什么？",
        # "抗癌肽与传统化疗药物相比有什么优势？",
    ]

    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 80)
        try:
            result = await enhanced_rag_agent.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })
            final_message = result['messages'][-1]
            if hasattr(final_message, 'content'):
                print(f"💬 回答:\n{final_message.content}")
            else:
                print(f"⚠️ 回答类型: {type(final_message)}")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
        print("=" * 80)


if __name__ == "__main__":
    print("\n🚀 增强版抗癌肽RAG Agent 已启动（LlamaIndex版本）")
    print("📊 使用 LlamaIndex FAISS 向量数据库")
    print("📁 数据库路径: mcp_course_materials_db_llamaindex/")
    print("🔧 嵌入模型: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n")

    # 运行测试
    asyncio.run(test_enhanced_rag_agent())
