import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

FEEDBACK_PROMPT = """You are an expert technical interview coach analyzing a mock interview session.

Given the full interview transcript, generate a structured feedback report.

Your report must follow this exact format:

OVERALL SCORE: X/10

TOPICS COVERED:
- topic 1
- topic 2

STRENGTHS:
- strength 1
- strength 2

GAPS IDENTIFIED:
- gap 1
- gap 2

IMPROVEMENT SUGGESTIONS:
- suggestion 1
- suggestion 2

SUMMARY:
2-3 sentence overall summary of the candidate's performance.

Be honest, specific, and constructive. Reference actual things the candidate said.
Do not be encouraging or soft. Be direct like a senior engineer would be."""


class FeedbackGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

    def generate(self, conversation_history: list, domain: str, difficulty: str) -> str:
        transcript = self._format_transcript(conversation_history)

        prompt = f"""Domain: {domain}
Difficulty: {difficulty}

Interview Transcript:
{transcript}

Generate the feedback report now."""

        messages = [
            SystemMessage(content=FEEDBACK_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _format_transcript(self, conversation_history: list) -> str:
        lines = []
        for msg in conversation_history:
            role = "Interviewer" if msg.__class__.__name__ == "AIMessage" else "Candidate"
            lines.append(f"{role}: {msg.content}")
        return "\n\n".join(lines)

    def print_report(self, report: str):
        print("\n" + "="*50)
        print("         INTERVIEW FEEDBACK REPORT")
        print("="*50)
        print(report)
        print("="*50 + "\n")


if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage

    sample_history = [
        AIMessage(content="Can you explain how a hashmap works internally?"),
        HumanMessage(content="It stores key value pairs and uses a hash function to find the index."),
        AIMessage(content="How does it handle collisions when two keys map to the same index?"),
        HumanMessage(content="It uses chaining with linked lists to store multiple values at the same index."),
        AIMessage(content="Okay, let's move on."),
        AIMessage(content="What is the time complexity of inserting into a hashmap?"),
        HumanMessage(content="It is O of 1 on average but can be O of N in the worst case due to collisions."),
        AIMessage(content="Okay, let's move on."),
    ]

    generator = FeedbackGenerator()
    report = generator.generate(sample_history, domain="DSA", difficulty="Easy")
    generator.print_report(report)