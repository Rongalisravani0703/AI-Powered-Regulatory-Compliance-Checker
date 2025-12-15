
#  Milestone 1 — Load Dataset + Preprocess + Groq Integration

from groq import Groq
from dotenv import load_dotenv
import os
import re



#  Load API Key from .env

dotenv_path = os.path.join(os.getcwd(), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(" GROQ_API_KEY not found in .env file")

# --- OpenAI API Key setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # ensures LangChain sees it


# Load Dataset (.txt file)
dataset_path = os.path.join(os.getcwd(), "Data", "Dataset.txt")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"⚠ Dataset not found at {"C:/Users/ronga/Documents/GitHub/AI-Powered-Regulatory-Complian/Data/Dataset.txt"}")

with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read().strip()

print("Dataset loaded successfully!")
print(f"Characters in dataset: {len(text)}")


#  Preprocess Text (clean + split into chunks)
text = re.sub(r"\s+", " ", text)  # remove extra spaces/newlines
chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]  # split into 1000-char chunks
print(f"Preprocessed into {len(chunks)} chunks")


#  Integrate with Groq API (test query)
client = Groq(api_key=api_key)

try:
    print("\n Testing Groq API connection...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # supported Groq model
        messages=[
            {"role": "system", "content": "You are an AI assistant for regulatory compliance tasks."},
            {"role": "user", "content": "Summarize the key objectives of GDPR in 3 lines."}
        ],
        max_tokens=150,
        temperature=0.7
    )
    print("\n Groq API Response:\n")
    print(response.choices[0].message.content.strip())

except Exception as e:
    print(f" Error connecting to Groq API: {e}")
