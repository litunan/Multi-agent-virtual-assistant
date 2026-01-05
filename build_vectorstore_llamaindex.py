#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 LlamaIndex 构建 FAISS 向量数据库

与 LangChain 版本的对比：
- LangChain: DirectoryLoader + RecursiveCharacterTextSplitter
- LlamaIndex: SimpleDirectoryReader + SentenceSplitter

LlamaIndex 特点：
1. 更灵活的节点解析器（Node Parsers）
2. 内置的文档摘要和元数据提取
3. 支持层次化索引结构
4. 更好的中文分句支持
"""

import os
from pathlib import Path


def build_vectorstore_llamaindex():
    """使用 LlamaIndex 从 RAG_Document 文件夹构建向量数据库"""
    
    # 延迟导入，便于检查依赖
    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
        StorageContext,
        Settings,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    import faiss
    
    print("=" * 60)
    print("🦙 使用 LlamaIndex 构建向量数据库")
    print("=" * 60)
    
    # 1. 初始化本地 Embedding 模型
    print("\n📦 正在加载本地 Embedding 模型...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        normalize=True,
    )
    print("✅ Embedding 模型加载成功")
    
    # 2. 配置全局设置
    Settings.embed_model = embed_model
    Settings.chunk_size = 1000
    Settings.chunk_overlap = 200
    
    # 3. 加载文档
    print("\n📄 正在加载文档...")
    documents = SimpleDirectoryReader(
        input_dir="RAG_Document",
        recursive=True,  # 递归读取子目录
        required_exts=[".md"],  # 只读取 markdown 文件
        filename_as_id=True,  # 使用文件名作为文档ID
    ).load_data()
    print(f"✅ 已加载 {len(documents)} 个文档")
    
    # 显示加载的文档信息
    for doc in documents:
        print(f"   📄 {doc.metadata.get('file_name', 'unknown')}")
    
    # 4. 配置文本分割器（Node Parser）
    print("\n✂️ 正在分割文档...")
    node_parser = SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator=" ",  # 主要分隔符
        paragraph_separator="\n\n",  # 段落分隔符
        secondary_chunking_regex="[。！？；\n]",  # 中文句子分隔
    )
    
    # 解析文档为节点
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"✅ 文档已分割为 {len(nodes)} 个节点")
    
    # 5. 创建 FAISS 向量存储
    print("\n🔧 正在构建 FAISS 向量索引...")
    
    # 获取 embedding 维度
    sample_embedding = embed_model.get_text_embedding("test")
    embedding_dim = len(sample_embedding)
    print(f"   Embedding 维度: {embedding_dim}")
    
    # 创建 FAISS 索引
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    
    # 创建向量存储
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 6. 构建索引
    print("\n🏗️ 正在构建向量索引...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    
    # 7. 保存索引
    output_dir = "mcp_course_materials_db_llamaindex"
    print(f"\n💾 正在保存索引到 {output_dir}/...")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到磁盘
    index.storage_context.persist(persist_dir=output_dir)
    
    print("\n" + "=" * 60)
    print("✅ LlamaIndex 向量数据库构建完成！")
    print("=" * 60)
    print(f"\n📊 统计信息:")
    print(f"   - 文档数量: {len(documents)}")
    print(f"   - 节点数量: {len(nodes)}")
    print(f"   - Embedding 维度: {embedding_dim}")
    print(f"   - 存储位置: {output_dir}/")
    
    return index


def test_query(index=None):
    """测试查询功能"""
    from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import Settings
    from llama_index.llms.openai_like import OpenAILike
    import faiss
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config.load_key import load_key
    
    output_dir = "mcp_course_materials_db_llamaindex"
    
    # 初始化 LLM - 使用阿里云百炼 API
    print("\n🤖 正在初始化 LLM...")
    llm = OpenAILike(
        api_key=load_key("aliyun-bailian"),
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        is_chat_model=True,
    )
    Settings.llm = llm
    print("✅ LLM 初始化成功 (qwen-plus)")
    
    if index is None:
        print("\n🔍 正在加载已保存的索引...")
        
        # 重新初始化 embedding 模型
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="cpu",
            normalize=True,
        )
        Settings.embed_model = embed_model
        
        # 加载 FAISS 索引
        vector_store = FaissVectorStore.from_persist_dir(output_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=output_dir,
        )
        
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        print("✅ 索引加载成功")
    
    # 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # 返回前3个最相似的结果
        llm=llm,
    )
    
    # 测试查询
    test_queries = [
        "抗癌肽是什么？",
        "抗癌肽的作用机制有哪些？",
    ]
    
    print("\n" + "=" * 60)
    print("🧪 测试查询")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n❓ 查询: {query}")
        print("-" * 40)
        
        # 执行查询
        response = query_engine.query(query)
        
        print(f"📝 回答: {response.response[:500]}...")
        
        # 显示来源
        if response.source_nodes:
            print(f"\n📚 参考来源 ({len(response.source_nodes)} 个):")
            for i, node in enumerate(response.source_nodes, 1):
                score = node.score if hasattr(node, 'score') else 'N/A'
                file_name = node.metadata.get('file_name', '未知')
                print(f"   {i}. {file_name} (相关度: {score:.4f})")


def compare_with_langchain():
    """对比 LangChain 和 LlamaIndex 的分块结果"""
    print("\n" + "=" * 60)
    print("📊 LangChain vs LlamaIndex 对比")
    print("=" * 60)
    
    # 读取一个示例文档
    sample_file = "RAG_Document/001/001_updated.md"
    if not os.path.exists(sample_file):
        print(f"❌ 示例文件不存在: {sample_file}")
        return
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\n📄 示例文档: {sample_file}")
    print(f"   总长度: {len(content)} 字符")
    
    # LangChain 分块
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        langchain_chunks = langchain_splitter.split_text(content)
        print(f"\n🔗 LangChain 分块结果: {len(langchain_chunks)} 块")
        print(f"   平均块大小: {sum(len(c) for c in langchain_chunks) / len(langchain_chunks):.0f} 字符")
        
    except ImportError:
        print("\n⚠️ LangChain 未安装，跳过对比")
        langchain_chunks = []
    
    # LlamaIndex 分块
    try:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core import Document
        
        llamaindex_splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            secondary_chunking_regex="[。！？；\n]",
        )
        
        doc = Document(text=content)
        llamaindex_nodes = llamaindex_splitter.get_nodes_from_documents([doc])
        llamaindex_chunks = [node.get_content() for node in llamaindex_nodes]
        
        print(f"\n🦙 LlamaIndex 分块结果: {len(llamaindex_chunks)} 块")
        print(f"   平均块大小: {sum(len(c) for c in llamaindex_chunks) / len(llamaindex_chunks):.0f} 字符")
        
    except ImportError:
        print("\n⚠️ LlamaIndex 未安装，跳过对比")
        llamaindex_chunks = []
    
    # 显示前3块的对比
    if langchain_chunks and llamaindex_chunks:
        print("\n" + "-" * 40)
        print("前3块内容对比：")
        for i in range(min(3, len(langchain_chunks), len(llamaindex_chunks))):
            print(f"\n【第 {i+1} 块】")
            print(f"LangChain ({len(langchain_chunks[i])} 字符):")
            print(f"   {langchain_chunks[i][:100]}...")
            print(f"LlamaIndex ({len(llamaindex_chunks[i])} 字符):")
            print(f"   {llamaindex_chunks[i][:100]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # 对比模式
        compare_with_langchain()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 仅测试查询
        test_query()
    else:
        # 构建向量数据库并测试
        index = build_vectorstore_llamaindex()
        test_query(index)
