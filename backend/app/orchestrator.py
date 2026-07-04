from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

load_dotenv()

SYSTEM_PROMPT = """You are a strict but fair technical interviewer at a top tech company.
You are conducting a placement interview for a final-year computer science student.

Your behaviour rules:
- Ask one clear question at a time.
- After the student answers, evaluate internally on: correctness, depth, and clarity.
- If weak or incomplete: ask one targeted follow-up referencing something specific they said.
- If strong: say exactly "Okay, let's move on." and nothing else.
- Never explain, teach, or give away the answer.
- Never use encouraging words. Stay neutral.
- Keep responses under 3 sentences.
- Do not break character."""


class InterviewOrchestrator:
    def __init__(self, domain: str, difficulty: str = "Medium"):
        self.domain = domain
        self.difficulty = difficulty
        self.history = []
        self.questions_asked = []
        self.current_question_id = None
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.4,
        )

    def add_to_history(self, role: str, content: str):
        if role == "human":
            self.history.append(HumanMessage(content=content))
        else:
            self.history.append(AIMessage(content=content))

    def get_response(self, user_message: str) -> str:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(self.history)
        messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(messages)
        self.add_to_history("human", user_message)
        self.add_to_history("ai", response.content)
        return response.content

    def should_move_on(self, response: str) -> bool:
        move_on_phrases = ["let's move on", "move on", "next question", "okay.", "i see."]
        return any(phrase in response.lower() for phrase in move_on_phrases)

    def reset(self):
        self.history = []
        self.questions_asked = []
        self.current_question_id = None


def setup_session() -> dict:
    print("\n" + "="*50)
    print("   Welcome to AI Voice Mock Interviewer")
    print("="*50 + "\n")

    print("What would you like to practice today?")
    print("  1. DSA")
    print("  2. System Design (LLD)")
    print("  3. HR\n")

    domain_map = {"1": "DSA", "2": "System Design", "3": "HR"}
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in domain_map:
            domain = domain_map[choice]
            break
        print("Please enter 1, 2, or 3.")

    print(f"\nDifficulty level?")
    print("  1. Easy")
    print("  2. Medium")
    print("  3. Hard\n")

    difficulty_map = {"1": "Easy", "2": "Medium", "3": "Hard"}
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in difficulty_map:
            difficulty = difficulty_map[choice]
            break
        print("Please enter 1, 2, or 3.")

    print(f"\nGreat! Starting a {difficulty} {domain} interview session...\n")
    return {"domain": domain, "difficulty": difficulty}


if __name__ == "__main__":
    from audio_handler import AudioHandler
    from tts_handler import TTSHandler
    from retriever import QuestionRetriever

    print("Loading models, please wait...")
    retriever = QuestionRetriever()
    audio = AudioHandler()
    tts = TTSHandler()
    print("Ready!\n")

    session = setup_session()
    agent = InterviewOrchestrator(
        domain=session["domain"],
        difficulty=session["difficulty"]
    )

    first_q = retriever.get_question(
        domain=session["domain"],
        difficulty=session["difficulty"]
    )

    if not first_q:
        print("No questions found for this selection. Exiting.")
        audio.close()
        exit()

    agent.current_question_id = first_q["id"]
    agent.questions_asked.append(first_q["id"])

    opening = first_q["question"]
    print(f"\nInterviewer: {opening}\n")
    tts.speak(opening)
    agent.add_to_history("ai", opening)

    max_questions = 5
    questions_covered = 1

    while questions_covered <= max_questions:
        transcript = audio.record_answer()
        if not transcript.strip():
            print("  [no speech detected, please try again]\n")
            continue

        print(f"\nYou said: {transcript}\n")
        response = agent.get_response(transcript)
        print(f"Interviewer: {response}\n")
        tts.speak(response)

        if agent.should_move_on(response):
            questions_covered += 1
            if questions_covered > max_questions:
                break

            next_q = retriever.get_question(
                domain=session["domain"],
                difficulty=session["difficulty"]
            )

            if next_q and next_q["id"] not in agent.questions_asked:
                agent.questions_asked.append(next_q["id"])
                agent.current_question_id = next_q["id"]
                print(f"Interviewer: {next_q['question']}\n")
                tts.speak(next_q["question"])
                agent.add_to_history("ai", next_q["question"])
            else:
                closing = "That concludes our session. Well done."
                print(f"Interviewer: {closing}\n")