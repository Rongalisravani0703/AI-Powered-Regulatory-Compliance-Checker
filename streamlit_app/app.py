# streamlit_app/app.py
import os
import sys
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Ensure project root is on sys.path so imports resolve
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # AI-Powered-Regulatory-Complian
sys.path.append(str(PROJECT_ROOT))

# Import backend functions from your reg_compliance_system.py
from reg_compliance_system import (
    ensure_dirs,
    load_env_keys,
    milestone1_load_preprocess,
    milestone2_clause_extraction,
    run_qa_system,
    load_contracts,
    load_regulations,
    mock_fetch_regulation,
    scheduler,
    check_risks,
    apply_updates,
    show_suggestions,
    LLM,
    DATA_DIR,
    CONTRACTS_DIR,
    REGULATIONS_DIR,
    RESULTS_DIR,
    CLAUSE_CSV,
    CLAUSE_JSON,
    QA_INPUT_FILE,
    QA_OUTPUT_FILE,
)

# Page config
st.set_page_config(page_title="AI Regulatory Compliance System", layout="wide")
st.title("📘 AI-Powered Regulatory Compliance System")


# Ensure directories exist (backend function)
ensure_dirs()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📄 Upload Contract",
        "🔍 Clause Extraction",
        "⚠️ Risk Detection",
        "📑 Apply Regulatory Updates",
        "📝 Amendment Suggestions",
    ],
)

# Helper to safely read CSV/JSON result files
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# HOME
if page == "🏠 Home":
    st.header("Welcome 👋")
   

    # Counts
    num_contracts = len(list(Path(CONTRACTS_DIR).glob("*.txt")))
    num_regs = len(list(Path(REGULATIONS_DIR).glob("*.txt")))
    st.metric("Contracts detected", num_contracts)
    st.metric("Regulation files detected", num_regs)


# UPLOAD CONTRACT
if page == "📄 Upload Contract":
    st.header("Upload a Contract (.txt)")
    uploaded = st.file_uploader("Choose a .txt contract file", type=["txt"])
    if uploaded:
        save_path = Path(CONTRACTS_DIR) / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved: {save_path}")
        st.write("Current contracts:")
        st.write([p.name for p in Path(CONTRACTS_DIR).glob("*.txt")])

# CLAUSE EXTRACTION
if page == "🔍 Clause Extraction":
    st.header("Run Milestone 1 & 2: Dataset check + Clause extraction")

    groq_key, openai_key = load_env_keys()
    llm = LLM(groq_key, openai_key)

    if st.button("Dataset load & LLM test"):
        with st.spinner("Runniing Dataset..."):
            res = milestone1_load_preprocess(llm)
        if not res.get("loaded"):
            st.warning("Dataset.txt not found in Data/ — .")
        else:
            st.success(f"Dataset loaded ({res.get('chars',0)} chars); {len(res.get('chunks',[]))} chunks.")
            if res.get("llm_test"):
                st.write("LLM test response (trimmed):")
                st.code(res["llm_test"][:1000])

    if st.button("Clause extraction & risk"):
        with st.spinner("Running clause extraction..."):
            clauses = milestone2_clause_extraction(llm)
        if not clauses:
            st.info("No contracts found or no clauses extracted.")
        else:
            st.success(f"Extracted {len(clauses)} clause items.")
            st.dataframe(pd.DataFrame(clauses))
            # show saved CSV / JSON if present
            df = safe_read_csv(CLAUSE_CSV)
            if df is not None:
                st.markdown("Saved clause CSV:")
                st.dataframe(df)
                st.download_button("Download clause CSV", df.to_csv(index=False).encode("utf-8"), "risk_assessment.csv")
            j = safe_read_json(CLAUSE_JSON)
            if j is not None:
                st.markdown("Saved clause JSON (preview):")
                st.json(j)

# RISK DETECTION
if page == "⚠️ Risk Detection":
    st.header("Scan contracts for regulation keyword risks")
    contracts = load_contracts()
    regs = load_regulations()
    st.write(f"Loaded {len(contracts)} contracts and {len(regs)} regulations.")

    if st.button("Scan Now"):
        with st.spinner("Scanning..."):
            risks = check_risks(contracts, regs)
        # Show results in table form
        flat = []
        for cid, matches in risks.items():
            if matches:
                for m in matches:
                    flat.append({"contract": cid, "reg_id": m["reg_id"], "keyword": m["keyword"], "reg_title": m.get("reg_title")})
            else:
                flat.append({"contract": cid, "reg_id": None, "keyword": None, "reg_title": None})
        df = pd.DataFrame(flat)
        st.dataframe(df)
        st.download_button("Download risk scan CSV", df.to_csv(index=False).encode("utf-8"), "risk_scan.csv")

# APPLY REGULATORY UPDATES
if page == "📑 Apply Regulatory Updates":
    st.header("Apply detected regulations to contract files (creates new versions)")
    contracts = load_contracts()
    regs = load_regulations()
    st.write(f"Contracts: {len(contracts)}  ·  Regulations: {len(regs)}")

    if st.button("Apply Updates Now"):
        with st.spinner("Applying updates..."):
            updated = apply_updates(contracts, regs)
        if not updated:
            st.info("No updates applied (no new regs or already applied).")
        else:
            st.success(f"Applied updates to {len(updated)} contracts.")
            st.json(updated)
            st.write("New contract files in Contracts folder:")
            st.write([p.name for p in Path(CONTRACTS_DIR).glob("*.txt")])

# AMENDMENT SUGGESTIONS
if page == "📝 Amendment Suggestions":
    st.header("Amendment suggestions (per regulation)")
    regs = load_regulations()
    if st.button("Generate Suggestions"):
        suggestions = show_suggestions(regs)
        if not suggestions:
            st.info("No regulations found.")
        else:
            st.json(suggestions)

# Footer
st.markdown("---")
st.write("If you see 'Dataset.txt not found' or 'No contracts', upload your own .txt contracts and regulations.")