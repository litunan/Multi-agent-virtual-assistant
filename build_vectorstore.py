"""构建 FAISS 向量数据库"""
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.load_key import load_key


def build_vectorstore():
    """从 RAG_Document 文件夹构建 FAISS 向量数据库"""
    
    # 初始化 embeddings
    embeddings = OpenAIEmbeddings(
        api_key=load_key("aliyun-bailian"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v1"
    )
    
    print("正在加载文档...")
    
    # 加载所有 markdown 文档
    loader = DirectoryLoader(
        "RAG_Document",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = loader.load()
    print(f"已加载 {len(documents)} 个文档")
    
    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"文档已分割为 {len(splits)} 个块")
    
    # 构建 FAISS 向量数据库
    print("正在构建向量数据库...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # 保存向量数据库
    vectorstore.save_local("mcp_course_materials_db")
    print("✅ 向量数据库已保存到 mcp_course_materials_db/")


if __name__ == "__main__":
    build_vectorstore()
