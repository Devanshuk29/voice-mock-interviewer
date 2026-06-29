from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """You are a strict but fair technical interviewer at a top tech company.
You are conducting a placement interview for a final-year computer science student.

Your behaviour rules:
- Ask one clear question at a time.
- After the student answers, evaluate internally on: correctness, depth, and clarity.
- If weak or incomplete: ask one targeted follow-up referencing something specific they said.
- If strong: say "Okay, let's move on." and nothing else — do not provide a new question yet.
- Never explain, teach, or give away the answer.
- Never use encouraging words. Stay neutral.
- Keep responses under 3 sentences.
- Do not break character."""

class InterviewOrchestrator:
    def __init__(self, domain: str, company: str = None, topics: list = None):
        self.domain = domain
        self.company = company
        self.topics = topics or []
        self.history = []
        self.question_count = 0
        self.max_questions = 8
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

    def is_session_complete(self) -> bool:
        return self.question_count >= self.max_questions

    def reset(self):
        self.history = []
        self.question_count = 0


if __name__ == "__main__":
    agent = InterviewOrchestrator(domain="DSA")
    opening = "Let's begin. Can you explain how a hashmap works internally?"
    print(f"\nInterviewer: {opening}\n")
    agent.add_to_history("ai", opening)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        response = agent.get_response(user_input)
        print(f"\nInterviewer: {response}\n")