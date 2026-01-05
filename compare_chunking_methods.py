#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 LangChain 和 LlamaIndex 两种分块方式的检索效果

测试维度：
1. 检索速度
2. 检索相关性
3. 上下文完整性
4. 最终答案质量
"""

import sys
import os
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.load_key import load_key


def test_langchain_retrieval(query: str, top_k: int = 3) -> Dict[str, Any]:
    """测试 LangChain 版本的检索"""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # 加载 embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 加载向量数据库
    vectorstore = FAISS.load_local(
        folder_path="mcp_course_materials_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    
    # 执行检索
    start_time = time.time()
    docs = vectorstore.similarity_search_with_score(query, k=top_k)
    retrieval_time = time.time() - start_time
    
    # 整理结果
    results = []
    for doc, score in docs:
        results.append({
            'content': doc.page_content,
            'score': float(score),
            'length': len(doc.page_content),
            'metadata': doc.metadata
        })
    
    return {
        'method': 'LangChain',
        'retrieval_time': retrieval_time,
        'results': results,
        'total_length': sum(r['length'] for r in results),
        'avg_chunk_size': sum(r['length'] for r in results) / len(results) if results else 0
    }


def test_llamaindex_retrieval(query: str, top_k: int = 3) -> Dict[str, Any]:
    """测试 LlamaIndex 版本的检索"""
    from llama_index.core import StorageContext, load_index_from_storage, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    
    # 加载 embeddings
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        normalize=True,
    )
    Settings.embed_model = embed_model
    
    # 加载向量数据库
    output_dir = "mcp_course_materials_db_llamaindex"
    vector_store = FaissVectorStore.from_persist_dir(output_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=output_dir,
    )
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    
    # 创建检索器
    retriever = index.as_retriever(similarity_top_k=top_k)
    
    # 执行检索
    start_time = time.time()
    nodes = retriever.retrieve(query)
    retrieval_time = time.time() - start_time
    
    # 整理结果
    results = []
    for node in nodes:
        results.append({
            'content': node.get_content(),
            'score': float(node.score) if hasattr(node, 'score') else 0.0,
            'length': len(node.get_content()),
            'metadata': node.metadata
        })
    
    return {
        'method': 'LlamaIndex',
        'retrieval_time': retrieval_time,
        'results': results,
        'total_length': sum(r['length'] for r in results),
        'avg_chunk_size': sum(r['length'] for r in results) / len(results) if results else 0
    }


def generate_answer_with_context(query: str, context: str, method: str) -> Dict[str, Any]:
    """使用检索到的上下文生成答案"""
    from langchain_openai import ChatOpenAI
    
    model = ChatOpenAI(
        api_key=load_key("aliyun-bailian"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
    )
    
    prompt = f"""你是一位资深的抗癌肽研究专家。请基于以下检索到的上下文信息，为用户提供专业、详细的解答。

用户问题：
{query}

检索上下文：
{context}

请提供专业的科学解答："""
    
    start_time = time.time()
    response = model.invoke([{"role": "user", "content": prompt}])
    generation_time = time.time() - start_time
    
    return {
        'method': method,
        'answer': response.content,
        'generation_time': generation_time,
        'answer_length': len(response.content)
    }


def compare_retrieval_quality(query: str):
    """对比两种方法的检索质量"""
    
    print("=" * 80)
    print(f"🔍 查询: {query}")
    print("=" * 80)
    
    # 测试 LangChain
    print("\n📦 测试 LangChain 检索...")
    langchain_result = test_langchain_retrieval(query, top_k=3)
    
    # 测试 LlamaIndex
    print("📦 测试 LlamaIndex 检索...")
    llamaindex_result = test_llamaindex_retrieval(query, top_k=3)
    
    # 对比检索速度
    print("\n" + "=" * 80)
    print("⚡ 检索速度对比")
    print("=" * 80)
    print(f"LangChain:  {langchain_result['retrieval_time']:.4f} 秒")
    print(f"LlamaIndex: {llamaindex_result['retrieval_time']:.4f} 秒")
    speed_winner = "LangChain" if langchain_result['retrieval_time'] < llamaindex_result['retrieval_time'] else "LlamaIndex"
    print(f"✅ 速度优胜: {speed_winner}")
    
    # 对比检索结果
    print("\n" + "=" * 80)
    print("📊 检索结果统计")
    print("=" * 80)
    
    print(f"\n【LangChain】")
    print(f"  - 检索块数: {len(langchain_result['results'])}")
    print(f"  - 平均块大小: {langchain_result['avg_chunk_size']:.0f} 字符")
    print(f"  - 总上下文长度: {langchain_result['total_length']} 字符")
    print(f"  - 平均相关性分数: {sum(r['score'] for r in langchain_result['results']) / len(langchain_result['results']):.4f}")
    
    print(f"\n【LlamaIndex】")
    print(f"  - 检索块数: {len(llamaindex_result['results'])}")
    print(f"  - 平均块大小: {llamaindex_result['avg_chunk_size']:.0f} 字符")
    print(f"  - 总上下文长度: {llamaindex_result['total_length']} 字符")
    print(f"  - 平均相关性分数: {sum(r['score'] for r in llamaindex_result['results']) / len(llamaindex_result['results']):.4f}")
    
    # 显示检索内容预览
    print("\n" + "=" * 80)
    print("📄 检索内容预览（前3块）")
    print("=" * 80)
    
    for i in range(min(3, len(langchain_result['results']))):
        print(f"\n【第 {i+1} 块对比】")
        print(f"\nLangChain ({langchain_result['results'][i]['length']} 字符, 分数: {langchain_result['results'][i]['score']:.4f}):")
        print(f"  {langchain_result['results'][i]['content'][:200]}...")
        
        print(f"\nLlamaIndex ({llamaindex_result['results'][i]['length']} 字符, 分数: {llamaindex_result['results'][i]['score']:.4f}):")
        print(f"  {llamaindex_result['results'][i]['content'][:200]}...")
    
    # 生成答案对比
    print("\n" + "=" * 80)
    print("💬 生成答案对比")
    print("=" * 80)
    
    print("\n🤖 使用 LangChain 上下文生成答案...")
    langchain_context = "\n\n".join([r['content'] for r in langchain_result['results']])
    langchain_answer = generate_answer_with_context(query, langchain_context, "LangChain")
    
    print("🤖 使用 LlamaIndex 上下文生成答案...")
    llamaindex_context = "\n\n".join([r['content'] for r in llamaindex_result['results']])
    llamaindex_answer = generate_answer_with_context(query, llamaindex_context, "LlamaIndex")
    
    print(f"\n【LangChain 答案】({langchain_answer['answer_length']} 字符, 耗时: {langchain_answer['generation_time']:.2f}s)")
    print("-" * 80)
    print(langchain_answer['answer'])
    
    print(f"\n【LlamaIndex 答案】({llamaindex_answer['answer_length']} 字符, 耗时: {llamaindex_answer['generation_time']:.2f}s)")
    print("-" * 80)
    print(llamaindex_answer['answer'])
    
    # 综合评分
    print("\n" + "=" * 80)
    print("🏆 综合评分")
    print("=" * 80)
    
    print("\n【LangChain】")
    print(f"  ✅ 速度: {'快' if speed_winner == 'LangChain' else '慢'}")
    print(f"  📏 块大小: 较小 ({langchain_result['avg_chunk_size']:.0f} 字符)")
    print(f"  📚 上下文完整性: {'较差' if langchain_result['avg_chunk_size'] < 1000 else '良好'}")
    print(f"  🎯 适用场景: 精确匹配、短问题检索")
    
    print("\n【LlamaIndex】")
    print(f"  ✅ 速度: {'快' if speed_winner == 'LlamaIndex' else '慢'}")
    print(f"  📏 块大小: 较大 ({llamaindex_result['avg_chunk_size']:.0f} 字符)")
    print(f"  📚 上下文完整性: {'良好' if llamaindex_result['avg_chunk_size'] > 2000 else '一般'}")
    print(f"  🎯 适用场景: 复杂问题、需要完整上下文")
    
    print("\n" + "=" * 80)


def main():
    """主测试函数"""
    
    test_queries = [
        "抗癌肽是什么？",
        "抗癌肽的主要作用机制是什么？",
        "如何设计和优化抗癌肽的结构？",
    ]
    
    print("\n" + "=" * 80)
    print("🧪 LangChain vs LlamaIndex 检索效果对比测试")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"测试 {i}/{len(test_queries)}")
        print(f"{'='*80}")
        
        compare_retrieval_quality(query)
        
        if i < len(test_queries):
            input("\n按回车继续下一个测试...")


if __name__ == "__main__":
    main()
