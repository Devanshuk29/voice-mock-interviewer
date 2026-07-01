import json
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

QUESTIONS_DIR = Path(__file__).parent / "questions"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def load_questions(file_path: Path) -> list:
    with open(file_path, "r") as f:
        return json.load(f)

def ingest_all():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="interview_questions",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    question_files = ["dsa.json", "system_design.json", "hr.json"]
    total = 0

    for file_name in question_files:
        file_path = QUESTIONS_DIR / file_name
        questions = load_questions(file_path)

        ids = [q["id"] for q in questions]
        documents = [q["question"] for q in questions]
        metadatas = [
            {
                "domain": q["domain"],
                "topic": q["topic"],
                "subtopic": q["subtopic"],
                "difficulty": q["difficulty"],
                "companies": ", ".join(q["companies"]),
                "hints": " | ".join(q["strong_answer_hints"])
            }
            for q in questions
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Ingested {len(questions)} questions from {file_name}")
        total += len(questions)

    print(f"\nTotal questions in ChromaDB: {total}")
    print(f"Collection count: {collection.count()}")

if __name__ == "__main__":
    ingest_all()