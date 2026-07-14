import os
import random
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()


CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chroma_db")
)


class QuestionRetriever:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_collection(
                name="interview_questions",
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            print(f"  [retriever init error: {e}]")
            self.collection = None

    def get_question(self, domain: str, topic: str = None, difficulty: str = None, exclude_ids: list = None) -> dict:
        if not self.collection:
            return None

        try:
            if topic:
                where = {"$and": [{"domain": domain}, {"topic": topic}]}
            else:
                where = {"domain": domain}

            results = self.collection.query(
                query_texts=[f"{domain} {difficulty or ''} interview question"],
                n_results=30,
                where=where
            )

            candidates = []
            for i in range(len(results["ids"][0])):
                candidates.append({
                    "id": results["ids"][0][i],
                    "question": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i]
                })

            if exclude_ids:
                candidates = [c for c in candidates if c["id"] not in exclude_ids]

            if not candidates:
                print(f"  [no questions found for domain={domain}, difficulty={difficulty}]")
                return None

            if difficulty:
                difficulty_matches = [c for c in candidates if c["metadata"].get("difficulty") == difficulty]
                if difficulty_matches:
                    return random.choice(difficulty_matches)

            return random.choice(candidates)

        except Exception as e:
            print(f"  [retriever error: {e}]")
            return None

    def get_hints(self, question_id: str) -> list:
        if not self.collection:
            return []
        try:
            result = self.collection.get(ids=[question_id])
            if result and result["metadatas"]:
                hints_str = result["metadatas"][0].get("hints", "")
                return hints_str.split(" | ") if hints_str else []
            return []
        except Exception as e:
            print(f"  [hints error: {e}]")
            return []


if __name__ == "__main__":
    retriever = QuestionRetriever()

    print("Test 1 — DSA question:")
    q = retriever.get_question(domain="DSA", difficulty="Easy")
    if q:
        print(f"  Q: {q['question']}")
        print(f"  Difficulty: {q['metadata']['difficulty']}")

    print("\nTest 2 — LLD question:")
    q = retriever.get_question(domain="System Design", difficulty="Medium")
    if q:
        print(f"  Q: {q['question']}")
        print(f"  Difficulty: {q['metadata']['difficulty']}")

    print("\nTest 3 — HR question:")
    q = retriever.get_question(domain="HR", difficulty="Easy")
    if q:
        print(f"  Q: {q['question']}")
        print(f"  Subtopic: {q['metadata']['subtopic']}")