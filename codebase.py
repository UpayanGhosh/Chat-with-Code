from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()

project_path = input("Enter the path to the codebase: ")

source_files = []
with console.status("[bold]Scanning files…[/bold]", spinner="dots"):
    for root, __, files, in os.walk(project_path):
        for file in files:
            if file.lower().endswith((".py", ".ts", ".js", ".tsx", ".dart")):
                source_files.append(os.path.join(root, file))

documents = []
with Progress(
    SpinnerColumn(),
    TextColumn("{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    transient=True,
) as progress:
    task_id = progress.add_task("Reading source files…", total=len(source_files))
    for path in source_files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
        except (UnicodeDecodeError, OSError):
            progress.advance(task_id)
            continue
        documents.append(Document(page_content=code, metadata={"path": path}))
        progress.advance(task_id)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

with console.status("[bold]Splitting documents into chunks…[/bold]", spinner="dots"):
    chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name="codebase",
    persist_directory="./chroma_code_db",
    embedding_function=embeddings
)

BATCH_SIZE = 64
with Progress(
    SpinnerColumn(),
    TextColumn("{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    transient=True,
) as progress:
    task_id = progress.add_task("Embedding & indexing chunks…", total=len(chunks))
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vector_store.add_documents(batch)
        progress.advance(task_id, advance=len(batch))

retriever = vector_store.as_retriever(search_kwargs={"k": 10})