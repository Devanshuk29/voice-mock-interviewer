from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """You are a strict but fair technical interviewer at a top tech company.
You are conducting a placement interview for a final-year computer science student.

Your behaviour rules:
- Ask one clear question at a time. Never ask two questions together.
- After the student answers, evaluate their response internally.
- If the answer is weak, vague, or incomplete: ask one specific follow-up to probe the exact gap.
- If the answer is strong and complete: acknowledge briefly with "Okay." or "I see." and say you will move on.
- Never give away the answer or explain the concept. Only probe with questions.
- Never say "great", "good job", "excellent", or any encouraging words. Stay strictly neutral.
- Keep all your responses under 3 sentences.
- Do not break character under any circumstances."""

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
)

def get_response(conversation_history: list, user_message: str) -> str:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(conversation_history)
    messages.append(HumanMessage(content=user_message))
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    history = []
    opening = "Let's begin. Can you explain how a hashmap works internally?"
    print(f"\nInterviewer: {opening}\n")
    history.append(AIMessage(content=opening))

    student_answer = "It stores key-value pairs and uses a hash function to find the index."
    print(f"Student: {student_answer}\n")

    reply = get_response(history, student_answer)
    print(f"Interviewer: {reply}\n")