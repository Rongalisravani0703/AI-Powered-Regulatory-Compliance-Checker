"""
reg_compliance_system.py
Integrated System – Milestone 1, 2 & 3 (Backend)
- Clause extraction (LLM)
- Simple rule-based risk scoring
- Regulation tracking + contract update automation
- Safe JSON parsing for LLM outputs
- Uses fixed project paths for your workspace
"""

import os
import re
import json
import glob
import time
import threading
import ast
from uuid import uuid4
from datetime import datetime
import pandas as pd

# Embedding models (optional; used for Q/A)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None
    faiss = None

# Environment
from dotenv import load_dotenv

# LLM Clients (optional: will be None if packages/keys missing)
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =======================================================================================
#                                   GLOBAL CONFIG (FIXED PATHS)
# =======================================================================================
# NOTE: change this if you move the project folder
BASE_DIR = r"C:\Users\ronga\Documents\GitHub\AI-Powered-Regulatory-Complian"

DATA_DIR = os.path.join(BASE_DIR, "Data")
CONTRACTS_DIR = os.path.join(DATA_DIR, "Contracts")
REGULATIONS_DIR = os.path.join(DATA_DIR, "Regulations")

RESULTS_DIR = os.path.join(BASE_DIR, "Results")
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")

QA_INPUT_FILE = os.path.join(BASE_DIR, "qa_queries.txt")
QA_OUTPUT_FILE = os.path.join(RESULTS_DIR, "qa_answers.json")
CLAUSE_JSON = os.path.join(RESULTS_DIR, "risk_assessment.json")
CLAUSE_CSV = os.path.join(RESULTS_DIR, "risk_assessment.csv")

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
POLL_INTERVAL = 30


# =======================================================================================
#                                   SETUP
# =======================================================================================
def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(CONTRACTS_DIR, exist_ok=True)
    os.makedirs(REGULATIONS_DIR, exist_ok=True)


def load_env_keys():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    return os.getenv("GROQ_API_KEY"), os.getenv("OPENAI_API_KEY")


# =======================================================================================
#                                   UTILITIES
# =======================================================================================
def _safe_parse_json_like(text):
    """
    Try to parse text as JSON. Fallback to ast.literal_eval. Return None on failure.
    """
    if not text:
        return None
    t = text.strip()

    # Direct JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # Extract JSON array or object inside text
    try:
        m = re.search(r"(\{.*\}|\[.*\])", t, re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                pass
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass
    except Exception:
        pass

    # Replace single quotes with double quotes (best-effort)
    try:
        fixed = t.replace("'", "\"")
        return json.loads(fixed)
    except Exception:
        pass

    # Last resort
    try:
        return ast.literal_eval(t)
    except Exception:
        return None


def extract_json(text):
    """Safely pull a JSON array from LLM response; returns list or []."""
    parsed = _safe_parse_json_like(text)
    if isinstance(parsed, list):
        return parsed
    # if a single object provided, return as list wrapper
    if isinstance(parsed, dict):
        return [parsed]
    return []


def rule_based_risk(text):
    """Simple rule-based risk scoring. Returns (score:int, reasons:list)."""
    text = (text or "").lower()
    score = 0
    reasons = []

    if "consent" not in text:
        score += 40
        reasons.append("Missing consent")

    if "indefinite" in text and "retain" in text:
        score += 30
        reasons.append("Indefinite data retention")

    if "sensitive" in text or "special category" in text:
        score += 20
        reasons.append("Sensitive data involved")

    if "lawful" in text:
        score = max(0, score - 10)

    return min(score, 100), reasons


# =======================================================================================
#                                   LLM WRAPPER
# =======================================================================================
class LLM:
    """
    Simple wrapper: tries Groq (if configured) then OpenAI (if configured).
    If neither available, ask() returns None.
    """

    def __init__(self, groq_key=None, openai_key=None):
        self.groq = Groq(api_key=groq_key) if Groq and groq_key else None
        self.openai = OpenAI(api_key=openai_key) if OpenAI and openai_key else None

    def ask(self, prompt, max_tokens=700):
        """Return string response or None."""
        # Try Groq
        if self.groq:
            try:
                resp = self.groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                return getattr(resp.choices[0].message, "content", None)
            except Exception:
                pass

        # Try OpenAI
        if self.openai:
            try:
                resp = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                return resp.choices[0].message.content
            except Exception:
                pass

        return None


# =======================================================================================
#                                  MILESTONE 1
# =======================================================================================
def milestone1_load_preprocess(llm):
    """
    Load Data/Dataset.txt (if present), return details dict:
    { loaded: bool, chars: int, chunks: [...], llm_test: str|None }
    """
    dataset_path = os.path.join(DATA_DIR, "Dataset.txt")
    result = {"loaded": False, "chars": 0, "chunks": [], "llm_test": None}

    if not os.path.exists(dataset_path):
        return result

    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        text = re.sub(r"\s+", " ", f.read()).strip()

    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    result["loaded"] = True
    result["chars"] = len(text)
    result["chunks"] = chunks

    # Optional LLM test
    try:
        resp = llm.ask("Explain the main objectives of GDPR in three lines.")
        result["llm_test"] = resp
    except Exception:
        result["llm_test"] = None

    return result


# =======================================================================================
#                                  MILESTONE 2
# =======================================================================================
def milestone2_clause_extraction(llm):
    """
    Load all .txt contracts under Data/Contracts, ask LLM to extract clauses,
    apply rule-based risk scoring, save CSV/JSON results and return enriched list.
    """
    files = [f for f in os.listdir(CONTRACTS_DIR) if f.lower().endswith(".txt")]
    if not files:
        return []

    # Read and join a limited context for LLM prompt
    docs = []
    for fname in files:
        fp = os.path.join(CONTRACTS_DIR, fname)
        try:
            docs.append(open(fp, "r", encoding="utf-8").read())
        except Exception:
            docs.append("")

    text = "\n\n".join(docs)

    prompt = f"""
Extract key legal clauses from the following contract text and return a JSON array like:
[
  {{"Clause": "Data Protection", "Reason": "Mentions personal data handling and consent"}}
]
Text:
{text[:4000]}
"""

    raw = llm.ask(prompt) or ""
    clauses = extract_json(raw)

    enriched = []
    for c in clauses:
        clause = c.get("Clause", "") if isinstance(c, dict) else ""
        reason = c.get("Reason", "") if isinstance(c, dict) else ""
        score, reasons = rule_based_risk(clause)
        enriched.append({
            "clause": clause,
            "reason": reason,
            "risk_score": score,
            "risk": "High" if score >= 60 else "Medium" if score >= 30 else "Low",
            "action": "Immediate" if score >= 80 else "Review" if score >= 50 else "LowPriority"
        })

    # Save outputs if any
    if enriched:
        ensure_dirs()
        try:
            pd.DataFrame(enriched).to_csv(CLAUSE_CSV, index=False, encoding="utf-8-sig")
        except Exception:
            # if pandas fails, ignore saving
            pass
        try:
            with open(CLAUSE_JSON, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return enriched


def run_qa_system(clauses, llm):
    """
    Automated file-driven Q/A:
    - Reads qa_queries.txt at project root
    - Builds small embedding index using sentence-transformers (if available)
    - Uses LLM to answer questions based on top-k contexts
    Returns list of {question, answer}
    """
    if not clauses:
        return []

    if not os.path.exists(QA_INPUT_FILE):
        return []

    if SentenceTransformer is None or faiss is None:
        # Embedding stack unavailable — fallback to calling LLM per question with full context
        questions = [q.strip() for q in open(QA_INPUT_FILE, "r", encoding="utf-8") if q.strip()]
        results = []
        context = "\n\n".join([c["clause"] + " - " + c.get("reason", "") for c in clauses])
        for q in questions:
            prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer in 3-4 sentences."
            ans = llm.ask(prompt) or "No response."
            results.append({"question": q, "answer": ans})
        ensure_dirs()
        with open(QA_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    # Build embedding index
    questions = [q.strip() for q in open(QA_INPUT_FILE, "r", encoding="utf-8") if q.strip()]
    if not questions:
        return []

    embedder = SentenceTransformer(EMBED_MODEL)
    corpus = [c["clause"] + " " + c.get("reason", "") for c in clauses]
    vectors = embedder.encode(corpus, show_progress_bar=False).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    qa_results = []
    for q in questions:
        q_vec = embedder.encode([q]).astype("float32")
        _, ids = index.search(q_vec, TOP_K)
        ctx = "\n".join([corpus[i] for i in ids[0] if i < len(corpus)])
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer in 3-5 sentences."
        ans = llm.ask(prompt) or "No response available."
        qa_results.append({"question": q, "answer": ans})

    ensure_dirs()
    with open(QA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=2, ensure_ascii=False)

    return qa_results


# =======================================================================================
#                                  MILESTONE 3 (Regulation tracking + updates)
# =======================================================================================
def load_contracts():
    """Return dict of contracts keyed by id (filename without .txt)."""
    contracts = {}
    if not os.path.exists(CONTRACTS_DIR):
        return contracts

    for fp in glob.glob(os.path.join(CONTRACTS_DIR, "*.txt")):
        cid = os.path.basename(fp).replace(".txt", "")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = ""
        contracts[cid] = {
            "id": cid,
            "title": os.path.basename(fp),
            "content": content,
            "version": 1,
            "applied_regulations": []
        }
    return contracts


def load_regulations():
    """Return list of regulation objects with auto-extracted keywords."""
    regs = []
    if not os.path.exists(REGULATIONS_DIR):
        return regs

    for fp in glob.glob(os.path.join(REGULATIONS_DIR, "*.txt")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                t = f.read().lower()
        except Exception:
            t = ""
        rid = os.path.basename(fp).replace(".txt", "")
        keywords = [w for w in ["encryption", "audit", "monitoring", "notice", "privacy", "consent", "data", "transfer", "localisation"] if w in t]
        regs.append({
            "id": rid,
            "title": os.path.basename(fp),
            "summary": t[:200],
            "keywords": keywords
        })
    return regs


def mock_fetch_regulation(regs):
    """Append a mock regulation to regs list and return it (for demo)."""
    new_reg = {
        "id": f"reg-{uuid4().hex[:6]}",
        "title": "Mock Privacy Update",
        "summary": "New transparency rule.",
        "keywords": ["privacy", "transparency"]
    }
    regs.append(new_reg)
    return new_reg


def scheduler(regs):
    """Background updater (daemon). Appends one mock regulation each interval."""
    while True:
        try:
            mock_fetch_regulation(regs)
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)


def check_risks(contracts, regs):
    """
    Check which regulations match each contract by keyword.
    Returns dict mapping contract_id -> list of {reg_id, keyword}.
    """
    result = {}
    for cid, c in (contracts or {}).items():
        text = (c.get("content") or "").lower()
        matches = []
        for r in (regs or []):
            for kw in r.get("keywords", []):
                if kw and kw in text:
                    matches.append({"reg_id": r["id"], "keyword": kw, "reg_title": r.get("title")})
        result[cid] = matches
    return result


def apply_updates(contracts, regs):
    """
    Apply updates to contract files: append clauses for regs not yet applied.
    Returns list of updated contract summaries.
    """
    updated = []
    for cid, c in (contracts or {}).items():
        updated_text = (c.get("content") or "") + "\n\n# Auto Added Clauses\n"
        newly_added = []
        for r in (regs or []):
            if r["id"] not in c.get("applied_regulations", []):
                updated_text += f"- Added clause for: {r['title']}\n"
                newly_added.append(r["id"])
                c.setdefault("applied_regulations", []).append(r["id"])

        if newly_added:
            c["version"] = c.get("version", 1) + 1
            new_file = os.path.join(CONTRACTS_DIR, f"{cid}_v{c['version']}.txt")
            try:
                with open(new_file, "w", encoding="utf-8") as f:
                    f.write(updated_text)
            except Exception:
                pass
            c["content"] = updated_text
            updated.append({"contract": cid, "new_version": c["version"], "added_regs": newly_added})
    return updated


def show_suggestions(regs):
    """
    Return a list of suggestion objects for each regulation:
    [{ 'reg_id':..., 'title':..., 'suggested_keywords':[...]}]
    """
    suggestions = []
    for r in (regs or []):
        suggestions.append({
            "reg_id": r.get("id"),
            "title": r.get("title"),
            "suggested_keywords": r.get("keywords", []),
        })
    return suggestions


# =======================================================================================
#                                  MAIN (CLI)
# =======================================================================================
def main():
    ensure_dirs()
    groq_key, openai_key = load_env_keys()
    llm = LLM(groq_key, openai_key)

    print("\nINTEGRATED COMPLIANCE SYSTEM (CLI)\n")

    regs = load_regulations()
    # start background scheduler to append mock regs (daemon)
    threading.Thread(target=scheduler, args=(regs,), daemon=True).start()

    while True:
        print("""
--- MENU ---
1. Run Milestone 1 (Dataset Load & LLM Test)
2. Run Milestone 2 (Clause Extraction + Risk + QA)
3. Scan Contract Risks
4. Apply Regulatory Updates
5. Show Amendment Suggestions
6. Exit
""")
        ch = input("Enter choice: ").strip()
        if ch == "1":
            res = milestone1_load_preprocess(llm)
            print("Milestone1 result:", res)
        elif ch == "2":
            clauses = milestone2_clause_extraction(llm)
            print("Clauses extracted:", clauses)
            if clauses:
                qa = run_qa_system(clauses, llm)
                print("QA Results:", qa)
        elif ch == "3":
            contracts = load_contracts()
            risks = check_risks(contracts, regs)
            print("Risks:", json.dumps(risks, indent=2))
        elif ch == "4":
            contracts = load_contracts()
            updated = apply_updates(contracts, regs)
            print("Updates applied:", updated)
        elif ch == "5":
            sugg = show_suggestions(regs)
            print("Suggestions:", json.dumps(sugg, indent=2))
        elif ch == "6":
            print("Exiting...")
            break
        else:
            print("Invalid option.\n")


if __name__ == "__main__":
    main()
