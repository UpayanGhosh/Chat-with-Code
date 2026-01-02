from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from codebase import retriever
from rich.console import Console

console = Console()

model = OllamaLLM(model="llama3.2")

template = """
You are a senior software engineer explaining a codebase.

Use ONLY the provided code context to answer.

Each code snippet includes its file path.
Explain how the system works step-by-step.

Code context:
{code}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("------------------------------------------------------------------------------")
    question = input("Enter your question about the codebase (or 'q' to quit): ")
    print("------------------------------------------------------------------------------")
    if question.lower() == 'q':
        break
    with console.status("[bold cyan]Retrieving relevant code…[/bold cyan]", spinner="dots"):
        code_chunks = retriever.invoke(question)

    with console.status("[bold green]Thinking…[/bold green]", spinner="dots"):
        result = chain.invoke({"code": code_chunks, "question": question})

    console.print(result)