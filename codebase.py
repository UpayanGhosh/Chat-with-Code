from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

project_path = input("Enter the path to the codebase: ")

source_files = []
for root, __, files, in os.walk(project_path):
    for file in files:
        if file.endswith((".py", ".ts", ".js", ".tsx")):
            source_files.append(os.path.join(root, file))

documents = []
for path in source_files:
    with open(path , 'r', encoding='utf-8') as f:
        code = f.read()
        documents.append(Document(page_content=code, metadata={"path": path}))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name="codebase",
    persist_directory="./chroma_code_db",
    embedding_function=embeddings
)
vector_store.add_documents(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 6})