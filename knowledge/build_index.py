import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from agent.llm import get_embedding_model
from config import AgentConfig


# Qdrant 配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "fitness_guide"

# 文档路径
DOC_PATH = "knowledge/fitness-form-guide.md"



def load_document(file_path: str) -> Document:
    """加载markdown文档"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(text=content, metadata={"source": file_path})


def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    """创建Qdrant collection（如果不存在）"""
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name not in collection_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection already exists: {collection_name}")


def build_index():
    """构建向量索引并存储到Qdrant"""
    # 初始化Qdrant客户端
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 创建collection
    create_qdrant_collection(client, COLLECTION_NAME)

    # 初始化嵌入模型
    embed_model = get_embedding_model()

    # 初始化Qdrant向量存储
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embed_model
    )

    # 创建存储上下文
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 加载文档
    doc = load_document(DOC_PATH)
    print(f"Loaded document: {DOC_PATH}")

    # 将文档转换为节点并存储
    from llama_index.core.node_parser import MarkdownNodeParser
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([doc])

    print(f"Parsed {len(nodes)} nodes from document")

    # 存入向量数据库
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print(f"Index built successfully! Total nodes: {len(nodes)}")

    return index


if __name__ == "__main__":
    build_index()
