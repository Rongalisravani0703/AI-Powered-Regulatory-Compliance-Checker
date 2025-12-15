
# Milestone 2 – Retrieval-Augmented Generation (RAG) System

# 1. Introduction
This milestone implements a Retrieval-Augmented Generation system that automatically identifies key clauses in contracts and assesses risks based on regulatory standards (GDPR dataset).

# 2. Concept of RAG
RAG combines:
-Retrieval – fetches relevant text chunks from a document knowledge base.
-Generation – uses a Large Language Model (LLM) to answer queries using retrieved context.

# 3. Architecture
1. Load regulatory and contract documents (GDPR, policies)
2. Split documents into smaller chunks
3. Convert text chunks into embeddings
4. Store embeddings in a FAISS vector database
5. Retrieve top-K similar chunks based on query
6. Pass retrieved text to LLM (OpenAI GPT)
7. Generate response + perform risk analysis

# 4. Tools and Libraries
-LangChain for RAG pipeline
-OpenAI API for embeddings and LLM
-FAISS for vector storage
-Python

# 5. Dataset
- `gdpr.txt`: contains GDPR regulatory articles
- `data_protection_policy.txt`: sample company policy to check compliance

# 6. Step-by-Step Implementation
- Step 1: Ensure API key
- Step 2: Load documents
- Step 3: Split into chunks
- Step 4: Create FAISS index
- Step 5: Build retriever
- Step 6: Create RAG chain
- Step 7: Display sources
- Step 8: Run end-to-end pipeline

# 7. Sample Queries
- “What are the GDPR clauses related to data retention?”
- “Identify compliance risks in this company’s data processing policy.”

# 8. Results
- The system retrieves relevant GDPR articles.
- The LLM extracts key clauses and assigns a risk level (low/medium/high).

# 9. Future Enhancements
- Integrate risk scoring using ML models.
- Add more datasets like privacy compliance and legal contracts.


#  Set your OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = Your_API_Key_here
chunk_size = 1000
chunk_overlap = 200


"""
rag.py — End-to-end Retrieval-Augmented Generation pipeline for contract clause extraction
and risk assessment 
"""
## Access environment Variables(like API keys) 
import os
## Handle JSON data
import json
## Optional, used forany numeric ops
import math
## Easy file handling(instead of plain strings)
from pathlib import Path
## Type hints for better readability 
from typing import List, Dict, Any

# DocumentLoadres read files like(.txt,.pdf,.docx)
from langchain.document_loaders import DirectoryLoader,TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
# Breaks big text into smaller pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Convert text into numeric vectors for semantic search
from langchain.embeddings import OpenAIEmbeddings
# A vector database that stores embeddings for fast retrieval
from langchain.vectorstores import FAISS
# The LLM interface
from langchain.llms import OpenAI
# Defines how the LLM should recieve input
from langchain.prompts import PromptTemplate
# Combines retriever + LLM for Q&A
from langchain.chains import RetrievalQA

# ---------- Configuration ----------
DATA_DIR = Path("./data/contracts")
FAISS_INDEX_DIR = Path("./faiss_index")
CHUNK_SIZE = 1000           # tokens per chunk (tune)
CHUNK_OVERLAP = 200         # overlap tokens
TOP_K = 6                   # number of retrieved chunks
MODEL_CONTEXT_TOKENS = 131072  # e.g., 128k + some headroom; change for your LLM
LLM_MODEL_NAME = "gpt-4o-mini"  # example; change as available
TEMPERATURE = 0.0

# ---------- Step 1: Ensure API Key ----------
def ensure_api_keys():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable. Set it before running.")
    # Optional: check Google creds for Drive loader
    # if you plan to use Google Drive, ensure GOOGLE_APPLICATION_CREDENTIALS is set.


# ---------- Step 2: Find & load documents ----------
def load_documents(data_dir: Path) -> List:
    """
    Load documents from a local directory. Supports .txt, .pdf, .docx (via UnstructuredWordDocumentLoader).
    DirectoryLoader can be used too, but we handle file types for clearer metadata.
    """
    docs = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found. Create and put contract files inside.")
    for file in sorted(data_dir.iterdir()):
        if file.is_dir():
            continue
        fname = file.name.lower()
        try:
            if fname.endswith(".txt"):
                loader = TextLoader(str(file), encoding="utf-8")
                docs.extend(loader.load())
            elif fname.endswith(".pdf"):
                loader = PyPDFLoader(str(file))
                docs.extend(loader.load())
            elif fname.endswith(".docx") or fname.endswith(".doc"):
                loader = UnstructuredWordDocumentLoader(str(file))
                docs.extend(loader.load())
            else:
                # fallback - DirectoryLoader behavior
                print(f"[WARN] Unsupported extension for {file.name} — skipping.")
        except Exception as e:
            print(f"[ERROR] Failed to load {file.name}: {e}")
    print(f"[INFO] Loaded {len(docs)} documents from {data_dir}")
    # Ensure each document has metadata['source']
    for d in docs:
        if "source" not in d.metadata:
            d.metadata["source"] = getattr(d, "metadata", {}).get("file_path", "unknown")
    return docs

# ---------- Step 3: Split documents into chunks ----------
def split_documents(docs: List, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ".", "!", "?"]
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks

# ---------- Step 4: Make / Load FAISS vector store ----------
def create_or_load_faiss(chunks: List, persist_dir: Path) -> FAISS:
    embeddings = OpenAIEmbeddings()
    if persist_dir.exists():
        print("[INFO] Loading existing FAISS index from disk...")
        vect = FAISS.load_local(str(persist_dir), embeddings)
    else:
        print("[INFO] Building FAISS index (this may take a while)...")
        vect = FAISS.from_documents(chunks, embeddings)
        persist_dir.mkdir(parents=True, exist_ok=True)
        vect.save_local(str(persist_dir))
        print(f"[INFO] FAISS index saved to {persist_dir}")
    return vect

# ---------- Step 5: Build the retriever ----------
def build_retriever(vect: FAISS, top_k: int = TOP_K):
    retriever = vect.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    print(f"[INFO] Retriever built (top_k={top_k})")
    return retriever

# ---------- Simple rule-based risk assessor (stub, extendable) ----------
def assess_risk_for_clause(clause_text: str) -> Dict[str, Any]:
    """
    Very simple heuristic rules to illustrate the risk engine.
    Extend with domain rules and ML model later.
    Returns: {risk_level: 'low'|'medium'|'high', reasons: [ ... ]}
    """
    text = clause_text.lower()
    reasons = []
    score = 0.0

    # Example rule: unlimited liability
    if "unlimited liability" in text or "no cap" in text or "aggregate liability is unlimited" in text or "liability shall not be limited" in text:
        reasons.append("No limitation of liability / unlimited liability phrase detected")
        score += 0.9

    # Missing termination rights
    if "terminate" in text and ("only" in text or "for cause" in text) and "for convenience" not in text:
        reasons.append("Termination rights appear restrictive or only for cause")
        score += 0.5

    # Data processing without retention/security language
    if "personal data" in text or "data processing" in text:
        if "retain" not in text and "retention" not in text:
            reasons.append("No data retention period specified")
            score += 0.6
        if "secure" not in text and "security" not in text and "encryption" not in text:
            reasons.append("No security obligations found")
            score += 0.4

    # Confidentiality missing
    if "confidential" in text and ("shall maintain" not in text and "shall not disclose" not in text):
        reasons.append("Confidentiality clause lacks explicit non-disclosure obligations")
        score += 0.4

    # Heuristic thresholds
    if score >= 0.8:
        level = "high"
    elif score >= 0.4:
        level = "medium"
    else:
        level = "low"
    return {"risk_level": level, "risk_score": round(min(score, 1.0), 2), "reasons": reasons}

# ---------- Step 6: Create the RAG chain (prompt + LLM + stuffing) ----------
def build_rag_chain(retriever, model_name=LLM_MODEL_NAME, temperature=TEMPERATURE):
    prompt_template = """You are an expert contract analyst.
Given the retrieved CONTEXT passages and the QUESTION, extract all key clauses (clause_type and full_text) found relevant to the question.
For each clause, assess compliance risk using these categories: low|medium|high and provide short reasons.
Return ONLY valid JSON: a list of objects with fields:
- clause_type: one of [indemnity, termination, confidentiality, data_processing, liability, payment, governing_law, audit, other]
- text: the clause text (trimmed)
- source: source metadata (filename/page if available)
- risk: low|medium|high
- risk_reasons: list of short strings

QUESTION: {query}

CONTEXT:
{context}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])

    llm = OpenAI(model_name=model_name, temperature=temperature)  # deterministic extraction
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # stuffing retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print(f"[INFO] RAG chain built with model {model_name}")
    return qa

# ---------- Step 7: Pretty-print sources & parse JSON output ----------
def pretty_print_result(result: Dict):
    answer_text = result.get("result") or result.get("output_text") or ""
    print("\n=== RAG RAW ANSWER ===\n")
    print(answer_text)
    print("\n=== SOURCE SNIPPETS ===\n")
    sources = result.get("source_documents") or []
    for idx, doc in enumerate(sources, start=1):
        meta = getattr(doc, "metadata", {})
        source_name = meta.get("source", meta.get("file_path", "unknown"))
        snippet = getattr(doc, "page_content", "")[:600].strip()
        print(f"[{idx}] Source: {source_name}")
        print("---- snippet ----")
        print(snippet)
        print()

    # Try to parse JSON from answer_text
    parsed = None
    try:
        parsed = json.loads(answer_text.strip())
        print("[INFO] Successfully parsed JSON output from LLM.")
    except Exception:
        # Try to recover JSON inside text
        try:
            start = answer_text.find("[")
            end = answer_text.rfind("]") + 1
            if start != -1 and end != -1:
                raw = answer_text[start:end]
                parsed = json.loads(raw)
                print("[INFO] Extracted JSON array from LLM output.")
        except Exception as e:
            print("[WARN] Failed to parse JSON from LLM output:", e)

    if parsed:
        # Post-process: run the lightweight risk assessor to cross-check/augment
        for obj in parsed:
            text = obj.get("text", "")
            # If model didn't provide risk, compute via rule-hybrid
            if not obj.get("risk"):
                r = assess_risk_for_clause(text)
                obj["risk"] = r["risk_level"]
                obj["risk_reasons"] = r["reasons"]
            else:
                # still compute supplementary notes
                r = assess_risk_for_clause(text)
                obj.setdefault("risk_reasons", [])
                obj["risk_reasons"] = list(set(obj["risk_reasons"] + r["reasons"]))
        print("\n=== STRUCTURED CLAUSES (FINAL) ===\n")
        print(json.dumps(parsed, indent=2))
    else:
        print("[ERROR] Could not parse JSON from LLM output. Returning raw text.")

# ---------- Step 8: Glue it all together (main pipeline) ----------
def run_pipeline(data_dir=DATA_DIR, persist_dir=FAISS_INDEX_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, top_k=TOP_K):
    ensure_api_keys()
    check_token_budget(chunk_size, top_k, MODEL_CONTEXT_TOKENS)

    # Load
    docs = load_documents(data_dir)
    if not docs:
        print("[ERROR] No documents loaded. Put files in data/contracts/ and try again.")
        return

    # Split
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Vectorstore
    vect = create_or_load_faiss(chunks, persist_dir)

    # Retriever
    retriever = build_retriever(vect, top_k=top_k)

    # RAG chain
    qa_chain = build_rag_chain(retriever)

    # Interactive loop
    print("\n[READY] Enter queries. Type 'exit' or 'quit' to stop.")
    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        print("[INFO] Running RAG retrieval + generation...")
        result = qa_chain({"query": q})
        pretty_print_result(result)
