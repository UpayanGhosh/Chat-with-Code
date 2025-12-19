# Local AI Agents — a from-scratch investigative walkthrough (Ollama + LangChain + RAG)

In this article, we will reconstruct your project **as if we were building it from scratch**, one design decision at a time, until every line of your handwritten Python files feels inevitable.

The project title I’ll use throughout is **“Local AI Agents”**, and the core idea is deceptively simple:

> Ask questions about a codebase, and get answers grounded in the actual source.

But “simple” is exactly where the trap is.

At this point, you might be wondering:

- Why can’t we just paste the code into a prompt and ask the model?
- Why do we need embeddings at all?
- What is a vector store doing here?
- Where does RAG actually happen in code?
- If I removed a line, what would break?

We’ll answer all of that — and we’ll do it **by walking your code line-by-line**, but in an order that matches how a human engineer would discover the architecture.

## Scope (strict)

We will explain **only** your handwritten Python source files:

- `main.py`
- `codebase.py`

We will **not** explain virtual environments, dependency managers, installation steps, or OS/tooling boilerplate. We’ll focus exclusively on:

- logic
- control flow
- data flow
- architectural intent
- design reasoning

## The big picture (hold this mental model)

Before we move forward, let’s pause and install a mental model — because without it, the code will feel like “magic”.

Your program has two distinct phases:

1. **Indexing phase (offline-ish)**: turn a codebase into a searchable memory.
   - Read source files
   - Split them into chunks
   - Embed each chunk into a vector (math)
   - Store vectors in a vector database (Chroma)
   - Produce a `retriever` object

2. **Question-answering phase (interactive)**: turn a user question into an answer grounded in retrieved chunks.
   - Embed the user question
   - Similarity-search the vector DB
   - Retrieve the top-k relevant chunks
   - Construct a prompt containing those chunks
   - Ask a local LLM via Ollama

And here’s the key constraint that shapes everything:

> A language model does not “search” your codebase. It only continues text. If you want grounded answers, you must **retrieve** the right context and place it in the prompt.

That “retrieve then generate” loop is RAG.

We’ll repeat this end-to-end pipeline multiple times until it becomes second nature:

**user question → embedding → vector similarity search → retrieved chunks → prompt construction → LLM reasoning → final response**

Keep that sequence in your head; every import and variable will snap into place.

---

# Part 1 — The core problem

## 1.1 Why plain LLMs are insufficient for “explain my codebase”

Imagine we tried the naive approach:

- Take the entire repository
- Paste it into a prompt
- Ask: “Explain how this works”

Why does this fail?

1. **Context window**: the model can only “see” a limited number of tokens at once.
2. **Precision**: even if it sees the code, it may ignore important details.
3. **Grounding**: without constraints, it may “helpfully” invent explanations.

Your `main.py` explicitly tries to prevent that third failure mode (hallucination) with:

> “Use ONLY the provided code context to answer.”

But there’s a hidden issue: the instruction is only useful if we can reliably provide the right context.

So the real engineering problem becomes:

> How do we feed the model only the code it needs, without feeding the entire codebase?

That is a **retrieval** problem.

## 1.2 Why local models make retrieval even more important

Local models are great: private, predictable, cheap to run. But they tend to have smaller context windows and weaker general reasoning than frontier hosted models.

So you compensate by improving the one thing you control:

- **the evidence you provide**

Retrieval is how you turn “LLM guessing” into “LLM reading”.

---

# Part 2 — Starting from the interactive goal (`main.py`)

Let’s start where a human would start: with the user experience.

What do we want?

- I run a program.
- It asks me: “Enter your question about the codebase”.
- I ask something like: “Where does retrieval happen?”
- It answers in a grounded, step-by-step way.

Your `main.py` is that interface.

## 2.1 Reading `main.py` top-to-bottom

Here is the first line:

```python
from langchain_ollama.llms import OllamaLLM
```

### What it is
`OllamaLLM` is a **LangChain LLM wrapper** that talks to a locally running Ollama server.

### Where it comes from
- Package namespace: `langchain_ollama`
- Module: `.llms`
- Class: `OllamaLLM`

### Why we chose it
Because the project’s identity is “local agents”: the “LLM” here is not an OpenAI API call; it’s a local model served by Ollama.

### What alternatives exist
LangChain supports many LLM backends. Alternatives include (conceptually):

- other local backends (different providers/wrappers)
- hosted APIs (not local)

But: your design goal is “local + private”, so `OllamaLLM` fits.

### What would break if we removed it
Everything downstream that expects a `model` object capable of generating text would fail — specifically, `chain = prompt | model` would have no right-hand side.

---

Next import:

```python
from langchain_core.prompts import ChatPromptTemplate
```

### What it is
`ChatPromptTemplate` is a structured way to create prompts from templates with variables.

You might ask: “Why do we need a prompt template? Why not just use an f-string?”

Because we want:

- an explicit declaration of which variables exist (`{code}`, `{question}`)
- compatibility with LangChain’s pipeline composition (`prompt | model`)

### Alternatives
- plain Python string formatting
- other LangChain prompt template classes

### What would break if removed
You’d lose `ChatPromptTemplate.from_template(template)`, and your `chain` composition would need to change.

---

Third import:

```python
from codebase import retriever
```

This is the quiet, architecturally critical line.

### What it is
You are importing a single variable, `retriever`, from your own file `codebase.py`.

### Where it comes from
It’s created at the bottom of `codebase.py` by calling:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 6})
```

### Why we chose this design
You are drawing a boundary:

- `codebase.py` = “build a searchable memory of code”
- `main.py` = “use that memory to answer questions”

That separation matters because indexing is conceptually different from querying.

### The important gotcha (no magic)
In Python, **importing a module executes its top-level code**.

So when `main.py` runs and does `from codebase import retriever`, Python will:

1. load `codebase.py`
2. execute it from top to bottom
3. only then expose `retriever`

That means:

- the vector store is built (or opened) at import time
- the program prompts for `project_path` at import time

In other words: the “indexing phase” happens *before* your interactive loop starts.

If you removed this import, you’d remove retrieval from your system entirely — and the model would have no code context.

---

## 2.2 Creating the LLM instance

```python
model = OllamaLLM(model="llama3.2")
```

### What it is
An object that knows how to call Ollama and produce a completion.

### Why the `model=` argument exists
Ollama can serve multiple models. You must name which model to run.

### What would happen if we changed it
Changing the string changes which model Ollama is asked to run.

### What would break if removed
If `model` is undefined, the pipeline `prompt | model` cannot be formed.

---

## 2.3 The prompt template: turning “retrieved chunks” into a question

```python
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
```

This string is the *contract* between retrieval and generation.

Let’s unpack the two variables:

- `{code}`: whatever retrieval returns
- `{question}`: what the user typed

Now, the constraint:

> “Use ONLY the provided code context to answer.”

This is your anti-hallucination policy. But it only works if you provide the right context.

At this point, you might be wondering:

- What format is `{code}` actually in?
- Is it raw text? A list? A set of objects?

In your current program, `retriever.invoke(question)` returns something that will be inserted into `{code}`.

In practice, LangChain retrievers often return a `list[Document]`.

If it *is* a list of `Document` objects, then `{code}` may become a string representation of those objects when formatted into the prompt.

That can still work (because the model sees text), but it’s worth noting:

- the better you control serialization (e.g., joining chunk texts cleanly), the more predictable the prompt becomes

We’ll come back to this when we examine `codebase.py`.

---

## 2.4 Turning the template into a LangChain prompt object

```python
prompt = ChatPromptTemplate.from_template(template)
```

### What it is
A prompt “factory” that knows it expects variables named `code` and `question`.

### Why it matters
This is what allows you to later call:

```python
chain.invoke({"code": code_chunks, "question": question})
```

If you mismatch the keys (say you used `"context"` instead of `"code"`), prompt formatting would fail.

---

## 2.5 The chain operator: `prompt | model`

```python
chain = prompt | model
```

This is one of the most important “LangChain idioms” in your project.

### What it is
A pipeline composition operator.

You can read it as:

> “Take the prompt, format it with inputs, then feed the resulting text into the model.”

### Why it exists
It makes your control flow explicit:

- prompts are upstream
- models are downstream

### Alternatives
You could call the model wrapper directly with a formatted string.

But using a chain:

- standardizes the invoke interface
- makes future extension easy (you could insert a parser, a guardrail, etc.)

### What would break if removed
You’d need to manually do prompt formatting and model invocation.

---

## 2.6 The interactive loop: the system in motion

```python
while True:
    print("------------------------------------------------------------------------------")
    question = input("Enter your question about the codebase (or 'q' to quit): ")
    print("------------------------------------------------------------------------------")
    if question.lower() == 'q':
        break
    code_chunks = retriever.invoke(question)
    result = chain.invoke({"code": code_chunks, "question": question})
    print(result)
```

Let’s narrate this like a senior engineer sitting next to you.

### Step A — loop forever
`while True` means the program becomes an interactive REPL-like session.

If we removed the loop, you’d only answer one question and exit.

### Step B — accept user input
`input(...)` blocks until the user types.

### Step C — allow quitting
`if question.lower() == 'q': break` is your exit hatch.

### Step D — retrieval happens here

```python
code_chunks = retriever.invoke(question)
```

This is where RAG begins.

But notice the subtlety:

- you are not giving the question to the LLM yet
- you are first giving it to the retriever

Why do we even need this?

Because retrieval is how we transform:

> “a question in English”

into:

> “the subset of code text likely relevant to that question”

We’ll soon explain how `retriever` achieves that using embeddings + similarity search.

### Step E — generation happens here

```python
result = chain.invoke({"code": code_chunks, "question": question})
```

This is where the program finally asks the LLM.

### Step F — print the answer
`print(result)` shows the model output.

---

# Part 3 — The first conceptual roadblock (and why embeddings exist)

Let’s pretend we didn’t have `codebase.py` yet.

We have `main.py` with an interactive loop.

Now the roadblock:

> How do we choose which code to feed into `{code}`?

A human might answer: “Search for relevant files.”

Okay — but **search how**?

- keyword search?
- regex?
- full text search?

Those work for some tasks, but the questions you want to ask are often *semantic*:

- “Where do you build the retriever?”
- “Why does import execute code?”
- “What connects retrieval to the prompt?”

These are not always keyword-aligned.

This is why we need **embeddings**.

## 3.1 Embeddings from first principles

An embedding is a mapping:

- input: text
- output: a vector (a list of numbers)

The purpose is to encode “meaning” into geometry.

A simple analogy:

- Words and sentences become points in a high-dimensional space.
- Similar meanings become points that are close together.

Why can’t we search text directly?

Because “meaning similarity” is not the same as “string similarity”.

Embeddings give you a way to compute similarity numerically, typically using a metric like cosine similarity.

And that leads to the central trick:

> If we embed *both* (a) code chunks and (b) the user question into the same vector space, then we can retrieve the chunks whose vectors are closest to the question vector.

That is the retrieval half of RAG.

---

# Part 4 — Creating the indexing + retrieval subsystem (`codebase.py`)

Now we’re ready for the file that makes everything work.

Open `codebase.py`:

```python
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
```

We’ll go import by import, because imports reveal architecture.

## 4.1 `import os`

### What it is
Python’s standard library module for interacting with the operating system — specifically filesystem traversal.

### Why it’s needed
You need to walk a directory tree and collect file paths.

### Alternative
- `pathlib` for a more object-oriented style

### What would break if removed
You’d have no `os.walk` and no `os.path.join`; file discovery would fail.

---

## 4.2 `from langchain_core.documents import Document`

### What it is
A `Document` is LangChain’s standard container for a chunk of text plus metadata.

Conceptually:

- `page_content`: the actual text (your code)
- `metadata`: extra information you want to carry along (like the file path)

### Why we use it
Because downstream components (splitters, vector stores, retrievers) expect `Document` objects.

### What would happen if we didn’t
You’d either:

- lose metadata (like `path`) or
- be forced to invent your own structure and convert later

### What would break if removed
`documents.append(Document(...))` would fail.

---

## 4.3 `from langchain_text_splitters import RecursiveCharacterTextSplitter`

### What it is
A text splitter that breaks long documents into smaller overlapping chunks.

### Why do we even need splitting?
Because embeddings and retrieval work best when:

- each chunk is small enough to be “about one thing”
- each chunk is small enough to fit into prompts later

If you embed an entire file as one vector, retrieval becomes coarse and prompts become huge.

### Why “recursive character” splitting specifically
This splitter tries to split along sensible boundaries (conceptually: paragraphs, newlines, etc.) while still enforcing a chunk size.

### Alternatives
LangChain offers other splitters (token-based, language-aware, etc.).

### What would break if removed
You’d have no `splitter.split_documents(documents)`, so you’d feed entire files into the vector store.

---

## 4.4 `from langchain_ollama import OllamaEmbeddings`

This line is the bridge between “text” and “math”.

### What `langchain_ollama` is
A LangChain integration package that knows how to talk to Ollama.

### What `OllamaEmbeddings` does (conceptually)
It takes text input and asks Ollama’s embedding-capable model to produce vectors.

Under the hood (high-level):

- it sends the text to Ollama
- Ollama runs an embedding model
- it returns a vector of floats

(Exact HTTP endpoints and response fields are implementation details, but the conceptual contract is: *text in, vector out*.)

### Why embeddings are needed *at this point*
Because you are about to build a vector database. A vector database cannot index raw strings; it indexes vectors.

### Alternatives
LangChain has many embedding providers (local and hosted). The key is: they must output vectors compatible with your vector store.

### Why this one fits a local-LLM setup
- It keeps everything local.
- It aligns with your “Local AI Agents” theme.

### What would break if removed
`embeddings = OllamaEmbeddings(...)` would fail, and your vector store would have no way to embed either documents or queries.

---

## 4.5 `from langchain_chroma import Chroma`

### What it is
A LangChain vector store integration for Chroma.

Think of Chroma as:

- a database where each row is (vector, text, metadata)
- with a similarity search index

### Why it’s here
Because you need persistent, queryable storage for embeddings.

### Alternatives
Other vector stores exist. But Chroma is lightweight and fits local workflows.

### What would break if removed
You’d have nowhere to store vectors, and thus no retriever.

---

## 4.6 The indexing pipeline, line-by-line

### 4.6.1 Ask the user what to index

```python
project_path = input("Enter the path to the codebase: ")
```

This line does something subtle: it makes indexing interactive.

Remember what we said earlier about imports executing top-level code?

Because `main.py` imports `retriever` from `codebase.py`, this prompt will appear as soon as the program starts.

If you removed this line, `project_path` wouldn’t exist and file discovery would fail.

A design question you might ask:

> Should indexing happen at import time?

Your current design says “yes”: you build/open the index when the program starts.

---

### 4.6.2 Discover source files

```python
source_files = []
for root, __, files, in os.walk(project_path):
    for file in files:
        if file.endswith((".py", ".ts", ".js", ".tsx")):
            source_files.append(os.path.join(root, file))
```

Let’s unpack this carefully.

#### `source_files = []`
A list that will store full paths to code files.

If you removed this, you’d have nowhere to accumulate results.

#### `os.walk(project_path)`
Produces a stream of `(root, dirs, files)` tuples across the directory tree.

You used `__` to ignore the directories list — a Python convention meaning “I intentionally won’t use this.”

If you replaced `os.walk` with a non-recursive approach, you’d miss nested folders.

#### `file.endswith((...))`
Filters to languages you care about.

Why include `.ts`, `.js`, `.tsx`?

Because you want the assistant to answer questions about a mixed codebase, not just Python.

What would happen if you removed the filter?

- You’d embed everything (including binaries, lockfiles, etc.)
- Retrieval quality would drop
- Indexing time and DB size would grow

#### `os.path.join(root, file)`
Builds an OS-correct path.

---

### 4.6.3 Load files into `Document` objects

```python
documents = []
for path in source_files:
    with open(path , 'r', encoding='utf-8') as f:
        code = f.read()
        documents.append(Document(page_content=code, metadata={"path": path}))
```

#### Why `with open(...)`
It ensures the file handle is closed properly.

#### Why `encoding='utf-8'`
Source files are text; UTF-8 is the safest default.

#### Why wrap code in `Document`
Because the **metadata** is critical.

Your `main.py` prompt says:

> “Each code snippet includes its file path.”

That’s only possible if you keep paths attached to chunks. This `metadata={"path": path}` is how you do it.

If you removed the metadata, the model would see code without origin — and your answers would become harder to trust and debug.

---

### 4.6.4 Split documents into chunks

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
```

#### `chunk_size=800`
This is the maximum chunk length (in characters for this splitter).

Why not larger?

- Larger chunks dilute topical focus.
- Larger chunks consume more prompt context.

Why not smaller?

- Too small means you lose surrounding context.

#### `chunk_overlap=100`
Overlap is your protection against boundary cuts.

If an important function definition spans the boundary between two chunks, overlap makes it likely one chunk still contains enough context.

What would happen if overlap were zero?

- Retrieval might find a chunk missing the crucial line just outside it
- The LLM would answer with partial evidence

---

### 4.6.5 Create embeddings

```python
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
```

This line declares: “My embedding space is produced by this specific model.”

Why do we specify a separate embedding model?

Because *generation* models and *embedding* models serve different purposes:

- generation: produce fluent text
- embeddings: produce stable, semantically meaningful vectors

Could you use the same model for both?

Sometimes, but it’s not typical. Your code chooses a dedicated embedding model.

What would happen if we removed this line?

- `Chroma(..., embedding_function=embeddings)` would have nothing to call
- queries would not embed
- similarity search becomes impossible

---

### 4.6.6 Create / open the vector store (Chroma)

```python
vector_store = Chroma(
    collection_name="codebase",
    persist_directory="./chroma_code_db",
    embedding_function=embeddings
)
```

Let’s treat each argument like an architectural decision.

#### `collection_name="codebase"`
A logical namespace inside the vector store.

If you changed this, you’d be writing/reading from a different collection.

#### `persist_directory="./chroma_code_db"`
This makes the index persistent on disk.

Why does persistence matter?

- Without it, you’d rebuild embeddings every run.
- With it, you can reuse the database.

Your workspace contains `chroma_code_db/chroma.sqlite3`, which is consistent with Chroma persisting data locally.

#### `embedding_function=embeddings`
This is the key: it tells the vector store how to convert text to vectors.

Without an embedding function, the vector store can’t add documents meaningfully, and it can’t embed queries.

---

### 4.6.7 Add chunk documents into the vector store

```python
vector_store.add_documents(chunks)
```

This line is the “indexing write”.

You are saying:

- Take every chunk
- Compute its embedding vector
- Store (vector, text, metadata)

At this point you might be wondering:

> What happens if I run the program twice?

In many setups, re-adding the same documents can create duplicates unless you manage IDs or clear the collection.

Your current code does not explicitly de-duplicate. That’s not “wrong” for learning, but it is a behavior worth understanding.

If you removed this line, the DB would be empty and retrieval would return nothing useful.

---

### 4.6.8 Export a retriever

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 6})
```

This is the boundary object imported by `main.py`.

#### What a “retriever” is
A retriever is an object with a simple job:

> given a query, return the most relevant documents

By wrapping the vector store, it hides the details:

- embedding the query
- running similarity search
- returning top matches

#### `search_kwargs={"k": 6}`
This says: return the top 6 chunks.

Why 6?

There’s a trade-off:

- higher `k`: more evidence, but bigger prompt and more noise
- lower `k`: smaller prompt, but risk missing key context

If you changed `k` to 1, you’d get very focused but brittle context.

If you changed it to 50, you might swamp the model with irrelevant code.

If you removed this line, `main.py` would have nothing to import as `retriever`.

---

# Part 5 — Now we can finally define RAG (only when we’re ready)

We’ve earned the definition.

**Retrieval-Augmented Generation (RAG)** is the pattern:

1. **Retrieve** relevant context from an external memory (your vector store).
2. **Augment** the prompt with that context.
3. **Generate** an answer conditioned on the augmented prompt.

Why this is not “just prompting”:

- prompting alone assumes the model already “knows” your code
- RAG forces the model to *read* your code

In your project, the retrieval half lives in `codebase.py`, and the generation half lives in `main.py`.

---

# Part 6 — Wiring everything back together (end-to-end walkthrough)

Now let’s walk the full pipeline slowly, then faster, then with “what if removed” questions.

## 6.1 Slow walkthrough: question → answer

### Step 1 — user types a question
`main.py` collects `question` via `input()`.

### Step 2 — question is sent to the retriever

```python
code_chunks = retriever.invoke(question)
```

What happens conceptually inside `invoke`?

1. The retriever uses the same embedding model (`OllamaEmbeddings`) to embed the question.
2. It compares that query vector against stored chunk vectors.
3. It selects the top `k=6` nearest chunks.
4. It returns those chunks (as `Document` objects) with `page_content` and `metadata`.

This is **vector similarity search** in action.

### Step 3 — retrieved chunks become prompt context

```python
result = chain.invoke({"code": code_chunks, "question": question})
```

Conceptually:

1. `ChatPromptTemplate` substitutes `{code}` and `{question}`.
2. The resulting text is sent to `OllamaLLM`.
3. Ollama generates tokens and returns a final response string.

### Step 4 — you print the answer

```python
print(result)
```

And the loop repeats.

---

## 6.2 Fast walkthrough (you should be able to say this out loud)

**user question → retriever embeds question → Chroma similarity search → top-k chunk Documents → prompt formats `{code}` + `{question}` → Ollama generates answer → printed output**

If any link in that chain breaks, the system collapses.

---

# Part 7 — Investigative questions: “what would happen if we removed this?”

This is how senior engineers build intuition: by imagining failure modes.

## 7.1 What if we remove `chunk_overlap=100`?

Your chunks become disjoint.

Practical impact:

- retrieval might return a chunk missing the line that actually answers the question
- the LLM may produce confident but incomplete explanations

## 7.2 What if we remove `metadata={"path": path}`?

The model sees code without provenance.

Practical impact:

- your prompt instruction “Each code snippet includes its file path” becomes untrue
- debugging answers becomes harder because you can’t trace where evidence came from

## 7.3 What if we remove `Use ONLY the provided code context`?

The model becomes more willing to generalize and invent.

Practical impact:

- sometimes answers sound better
- but they become less trustworthy for “what does *this* code do?”

## 7.4 What if we remove `retriever.invoke(question)` and pass only the question?

Then you no longer have RAG.

You’d be using a plain LLM chatbot.

The program would still run, but it would no longer be anchored to your repository.

---

# Part 8 — A final recap (what you’ve built)

In this article, we built a mental model that matches your code exactly:

1. `codebase.py` constructs a retrieval subsystem:
   - filesystem → documents → chunks → embeddings → Chroma vector store → `retriever`
2. `main.py` constructs a generation subsystem:
   - prompt template → Ollama LLM → chain
3. Runtime loop:
   - question → retrieve chunks → format prompt → generate answer

And the core “inevitable” pipeline — the one you should remember — is:

**user question → embedding → vector similarity search → retrieved chunks → prompt construction → LLM reasoning → final response**

If you can explain that pipeline without looking at the code, then you don’t just know how this works — you understand why it works.
