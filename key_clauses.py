
import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

try:
    from groq import Groq
except:
    Groq = None

try:
    from openai import OpenAI
except:
    OpenAI = None


# ---------------- CONFIG ----------------
DATA_FOLDER = r"C:\Users\ronga\Documents\GitHub\AI-Powered-Regulatory-Complian\Data\Contracts"
RESULTS_JSON = "Results/risk_assessment.json"
RESULTS_CSV = "Results/risk_assessment.csv"
QA_INPUT_FILE = "qa_queries.txt"
QA_OUTPUT_FILE = "Results/qa_answers.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


# ---------------- UTILITIES ----------------
def ensure_dirs():
    os.makedirs("Results", exist_ok=True)
    os.makedirs("Outputs", exist_ok=True)


def load_env():
    load_dotenv()
    return os.getenv("GROQ_API_KEY"), os.getenv("OPENAI_API_KEY")


def extract_json_from_text(text):
    if not text:
        return []
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return []
    return []


def rule_based_risk_score(text):
    text = text.lower()
    score, reasons = 0, []
    if "consent" not in text:
        score += 40
        reasons.append("Missing consent reference")
    if "retain" in text and "indefinite" in text:
        score += 25
        reasons.append("Indefinite retention")
    if "sensitive" in text:
        score += 20
        reasons.append("Sensitive data involved")
    if "lawful" in text:
        score -= 10
    return min(100, score), reasons


def assess_compliance_flags_for_clauses(clauses):
    enriched = []
    for c in clauses:
        clause_text = c.get("Clause", "Unknown")
        score, reasons = rule_based_risk_score(clause_text)
        enriched.append({
            "clause": clause_text,
            "risk": "High" if score >= 60 else "Medium" if score >= 30 else "Low",
            "reason": c.get("Reason", "; ".join(reasons) or "Rule-based"),
            "risk_score": score,
            "action": "Immediate" if score >= 80 else "Review" if score >= 50 else "LowPriority",
            "flags": "no_roles_defined" if "controller" not in clause_text.lower() else "ok"
        })
    return enriched


# ---------------- LLM HANDLER ----------------
class LLM:
    def __init__(self, groq_key, openai_key):
        self.groq = Groq(api_key=groq_key) if groq_key and Groq else None
        self.openai = OpenAI(api_key=openai_key) if openai_key and OpenAI else None

    def generate(self, prompt):
        if self.groq:
            try:
                r = self.groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=800,
                )
                return r.choices[0].message.content
            except Exception as e:
                print("⚠️ Groq failed:", e)
        if self.openai:
            try:
                r = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=800,
                )
                return r.choices[0].message.content
            except Exception as e:
                print("⚠️ OpenAI failed:", e)
        return None


# ---------------- MAIN PIPELINE ----------------
def main():
    ensure_dirs()
    groq_key, openai_key = load_env()
    llm = LLM(groq_key, openai_key)

    # Load text files
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")]
    if not files:
        print("❌ No contract files found.")
        return

    docs = [open(os.path.join(DATA_FOLDER, f), encoding="utf-8").read() for f in files]
    context = "\n\n".join(docs)

    prompt = f"""
    Extract key legal clauses from the following contract text and return a JSON array like:
    [
      {{"Clause": "Data Protection", "Reason": "Mentions personal data handling and consent"}},
      {{"Clause": "Liability", "Reason": "Defines responsibility for breach of contract"}}
    ]
    Text:
    {context[:4000]}
    """

    print("🤖 Extracting key clauses...")
    raw_text = llm.generate(prompt)
    clauses = extract_json_from_text(raw_text)
    enriched_clauses = assess_compliance_flags_for_clauses(clauses)

    # Save clause & risk results
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched_clauses, f, indent=2, ensure_ascii=False)
    pd.DataFrame(enriched_clauses).to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

    # Display key sections
    print("\n=========================")
    print("🔹 SECTION 1: KEY CLAUSES")
    print("=========================")
    for c in enriched_clauses:
        print(f"Clause: {c['clause']}")
        print(f"Summary: {c['reason']}\n")

    print("\n===============================================")
    print("🔹 SECTION 2: POTENTIAL COMPLIANCE RISK ANALYSIS")
    print("===============================================")
    for c in enriched_clauses:
        print(f"Clause: {c['clause']}")
        print(f"Risk Level: {c['risk']}")
        print(f"Reason: {c['reason']}")
        print(f"Risk Score: {c['risk_score']}")
        print(f"Action: {c['action']}")
        print(f"Flags: {c['flags']}\n")

    # Run automated Q/A if qa_queries.txt exists
    if os.path.exists(QA_INPUT_FILE):
        run_qa_file_mode(enriched_clauses, llm)
    else:
        print("\n⚠️ No qa_queries.txt file found — skipping Q/A panel.")


# ---------------- AUTOMATED Q/A FROM FILE ----------------
def run_qa_file_mode(enriched_clauses, llm):
    print("\n===============================")
    print("💬 AUTOMATED Q/A PANEL (from file)")
    print("===============================")

    # Load questions
    with open(QA_INPUT_FILE, "r", encoding="utf-8") as f:
        questions = [q.strip() for q in f.readlines() if q.strip()]

    # Prepare FAISS index
    embedder = SentenceTransformer(EMBED_MODEL)
    corpus = [c["clause"] + " " + c["reason"] for c in enriched_clauses]
    vectors = embedder.encode(corpus, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    qa_results = []

    for question in questions:
        q_vec = embedder.encode([question]).astype("float32")
        _, idxs = index.search(q_vec, TOP_K)
        context = "\n".join([corpus[i] for i in idxs[0]])

        qa_prompt = f"""
        You are a compliance analyst.
        Context:
        {context}
        Question: {question}
        Answer clearly in 3–5 sentences.
        """

        answer = llm.generate(qa_prompt) or "No response available."
        qa_results.append({"question": question, "answer": answer})
        print(f"\nQ: {question}\nA: {answer}\n")

    # Save all Q&A results
    with open(QA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, indent=2, ensure_ascii=False)

    print(f"✅ All Q/A saved to {QA_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
