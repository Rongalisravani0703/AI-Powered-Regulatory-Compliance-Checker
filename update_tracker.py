
import os
import glob
import time
import threading
from datetime import datetime, timezone
from uuid import uuid4

# ===========================
# CONFIG
# ===========================

BASE_DIR = r"C:\Users\ronga\Documents\GitHub\AI-Powered-Regulatory-Complian\Data"
CONTRACTS_DIR = os.path.join(BASE_DIR, "Contracts")
REGULATIONS_DIR = os.path.join(BASE_DIR, "Regulations")
POLL_INTERVAL_SECONDS = 30


# ===========================
# LOAD EXISTING DATA
# ===========================

def load_contracts():
    """Load ALL user-provided contract text files."""
    contracts = {}

    for file_path in glob.glob(os.path.join(CONTRACTS_DIR, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        file_name = os.path.basename(file_path)
        contract_id = file_name.replace(".txt", "")

        contracts[contract_id] = {
            "id": contract_id,
            "title": file_name,
            "file_path": file_path,
            "version": 1,  
            "content": text,
            "applied_regulations": []
        }

    return contracts


def load_regulations():
    """Load ALL regulation text files and extract keywords automatically."""
    regulations = []

    for file_path in glob.glob(os.path.join(REGULATIONS_DIR, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        file_name = os.path.basename(file_path)
        reg_id = file_name.replace(".txt", "")

        # simple keyword extraction
        keywords = []
        for word in ["encryption", "monitoring", "audit", "notice"]:
            if word in text:
                keywords.append(word)

        regulations.append({
            "id": reg_id,
            "title": file_name,
            "summary": text[:200] + "...",
            "keywords": keywords
        })

    return regulations


# ===========================
# MOCK FETCHER (optional)
# ===========================

def mock_fetch_regulations(regulations):
    new_reg = {
        "id": f"reg-mock-{uuid4().hex[:6]}",
        "title": "Mock Transparency Rule",
        "summary": "Requires profiling transparency.",
        "keywords": ["privacy", "transparency"]
    }

    regulations.append(new_reg)
    return new_reg


# ===========================
# RISK CHECK
# ===========================

def check_risks(contracts, regulations):
    print("\n=== Risk Scan Report ===")

    for cid, c in contracts.items():
        text = c["content"].lower()

        print(f"\nContract: {cid}")
        risks = []

        for r in regulations:
            for kw in r["keywords"]:
                if kw in text:
                    risks.append(f"{r['id']} → keyword match: {kw}")

        if risks:
            for r in risks:
                print(" -", r)
        else:
            print(" No risks detected")

    print()


# ===========================
# APPLY UPDATES
# ===========================

def apply_updates(contracts, regulations):
    print("\n=== Applying Updates ===")

    for cid, c in contracts.items():
        updated_text = c["content"] + "\n\n# Auto-Added Clauses\n"

        for r in regulations:
            if r["id"] not in c["applied_regulations"]:
                updated_text += f"- Added clause for: {r['title']}\n"
                c["applied_regulations"].append(r["id"])

        # Save new version file
        new_version = c["version"] + 1
        new_file = os.path.join(CONTRACTS_DIR, f"{cid}_v{new_version}.txt")

        with open(new_file, "w", encoding="utf-8") as f:
            f.write(updated_text)

        c["version"] = new_version
        c["file_path"] = new_file
        c["content"] = updated_text

        print(f"Updated → {cid} to v{new_version}")

    print()


# ===========================
# SHOW SUGGESTIONS
# ===========================

def show_suggestions(regulations):
    print("\n=== Amendment Suggestions ===")

    for r in regulations:
        print(f"\n• {r['title']}")
        print(" → Add clauses on:", ", ".join(r["keywords"]))

    print()


# ===========================
# Scheduler
# ===========================

def scheduler_loop(regulations):
   def scheduler_loop():
    while True:
        reg = mock_fetch_regulation()
        # Remove the print — store silently
        if reg:
            save_regulation(reg)   # silently save
        
        time.sleep(5)  # run every 5 seconds


# ===========================
# MAIN
# ===========================

def main():
    contracts = load_contracts()
    regulations = load_regulations()

    threading.Thread(target=scheduler_loop, args=(regulations,), daemon=True).start()

    while True:
        print("""
===== Regulatory Update Tracking System =====
1. Fetch new regulations now
2. Scan contract risks
3. Apply regulatory updates
4. Show amendment suggestions
5. Exit
""")
        ch = input("Enter choice: ")

        if ch == "1":
            new_reg = mock_fetch_regulations(regulations)
            print("\nNew Regulation:", new_reg["id"])

        elif ch == "2":
            check_risks(contracts, regulations)

        elif ch == "3":
            apply_updates(contracts, regulations)

        elif ch == "4":
            show_suggestions(regulations)

        elif ch == "5":
            print("Exiting...")
            break

        else:
            print("Invalid Option\n")


if __name__ == "__main__":
    main()
