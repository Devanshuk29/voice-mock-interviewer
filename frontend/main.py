import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "app"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "rag"))

from orchestrator import InterviewOrchestrator
from retriever import QuestionRetriever
from audio_handler import AudioHandler
from tts_handler import TTSHandler
from feedback import FeedbackGenerator

st.set_page_config(
    page_title="AI Voice Mock Interviewer",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #0d0d0d !important;
        color: #e0e0e0 !important;
    }

    .stApp { background-color: #0d0d0d; }

    .terminal-box {
        background: #111;
        border: 0.5px solid #2a2a2a;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
    }

    .interviewer-msg { color: #00ff41; font-size: 13px; margin-bottom: 4px; }
    .candidate-msg { color: #aaaaaa; font-size: 13px; margin-bottom: 4px; }
    .label-green { color: #00ff41; font-size: 11px; }
    .label-gray { color: #555; font-size: 11px; }
    .error-msg { color: #ff4444; font-size: 12px; }
    .status-bar {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 0.5px solid #1f1f1f;
        margin-bottom: 1rem;
        font-size: 11px;
        color: #3c3c3c;
    }

    div.stButton > button {
        background: #111 !important;
        color: #00ff41 !important;
        border: 0.5px solid #2a2a2a !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        padding: 8px 20px !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        border-color: #00ff41 !important;
        background: #0a1a0a !important;
    }

    .stSelectbox > div > div {
        background: #111 !important;
        border: 0.5px solid #2a2a2a !important;
        color: #e0e0e0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 6px !important;
    }

    .feedback-section {
        background: #111;
        border: 0.5px solid #2a2a2a;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #e0e0e0;
    }

    .score-display {
        font-size: 48px;
        font-weight: 500;
        color: #00ff41;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        padding: 1rem 0;
    }

    .section-label {
        color: #00ff41;
        font-size: 11px;
        margin-bottom: 6px;
        letter-spacing: 0.1em;
    }

    hr { border-color: #1f1f1f !important; }
    .stSpinner > div { color: #00ff41 !important; }

    [data-testid="stMarkdownContainer"] p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    defaults = {
        "screen": "setup",
        "domain": None,
        "difficulty": None,
        "agent": None,
        "retriever": None,
        "audio": None,
        "tts": None,
        "feedback_gen": None,
        "conversation": [],
        "questions_covered": 0,
        "max_questions": 5,
        "session_complete": False,
        "feedback_report": None,
        "models_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_models():
    if not st.session_state.models_loaded:
        try:
            st.session_state.retriever = QuestionRetriever()
            st.session_state.audio = AudioHandler()
            st.session_state.tts = TTSHandler()
            st.session_state.feedback_gen = FeedbackGenerator()
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"failed to load models: {e}")
            st.stop()


def render_setup():
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <p style='color:#00ff41; font-size:11px; letter-spacing:0.2em;'>── AI VOICE MOCK INTERVIEWER ──</p>
        <p style='color:#555; font-size:11px; margin-top:4px;'>placement interview preparation system v1.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='terminal-box'>", unsafe_allow_html=True)
    st.markdown("<p class='label-green'>// select domain</p>", unsafe_allow_html=True)
    domain = st.selectbox("", ["DSA", "System Design (LLD)", "HR"], label_visibility="collapsed", key="domain_select")
    st.markdown("<br><p class='label-green'>// select difficulty</p>", unsafe_allow_html=True)
    difficulty = st.selectbox("", ["Easy", "Medium", "Hard"], label_visibility="collapsed", key="diff_select")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("▶  initialize session"):
        with st.spinner("loading models..."):
            load_models()

        st.session_state.domain = domain
        st.session_state.difficulty = difficulty
        st.session_state.agent = InterviewOrchestrator(domain=domain, difficulty=difficulty)

        try:
            first_q = st.session_state.retriever.get_question(
                domain=domain,
                difficulty=difficulty,
                exclude_ids=st.session_state.agent.questions_asked
            )

            if not first_q:
                st.error("no questions found for this selection. please try a different domain or difficulty.")
                return

            st.session_state.agent.questions_asked.append(first_q["id"])
            st.session_state.agent.add_to_history("ai", first_q["question"])
            st.session_state.conversation.append({
                "role": "interviewer",
                "text": first_q["question"]
            })
            st.session_state.tts.speak(first_q["question"])
            st.session_state.screen = "interview"
            st.rerun()

        except Exception as e:
            st.error(f"failed to start session: {e}")


def render_interview():
    domain = st.session_state.domain
    difficulty = st.session_state.difficulty
    q_num = st.session_state.questions_covered + 1
    max_q = st.session_state.max_questions

    st.markdown(f"""
    <div class='status-bar'>
        <span>session: {domain} / {difficulty}</span>
        <span>question {q_num} of {max_q}</span>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.conversation:
        if msg["role"] == "interviewer":
            st.markdown(f"""
            <div class='terminal-box'>
                <p class='label-green'>[interviewer]</p>
                <p class='interviewer-msg'>{msg['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif msg["role"] == "candidate":
            st.markdown(f"""
            <div class='terminal-box'>
                <p class='label-gray'>[you]</p>
                <p class='candidate-msg'>{msg['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif msg["role"] == "error":
            st.markdown(f"""
            <div class='terminal-box'>
                <p class='error-msg'>[system] {msg['text']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("🎤  record answer  (pause 5s to stop)"):
            try:
                with st.spinner("listening..."):
                    transcript = st.session_state.audio.record_answer()

                if not transcript or len(transcript.strip()) < 3:
                    fallback = "I didn't quite catch that. Could you please repeat your answer?"
                    st.session_state.conversation.append({
                        "role": "interviewer",
                        "text": fallback
                    })
                    st.session_state.tts.speak(fallback)
                    st.rerun()
                    return

                st.session_state.conversation.append({
                    "role": "candidate",
                    "text": transcript
                })

                try:
                    with st.spinner("interviewer thinking..."):
                        response = st.session_state.agent.get_response(transcript)
                except Exception as e:
                    response = "I'm having a brief technical issue. Could you elaborate on your last point?"
                    st.session_state.conversation.append({
                        "role": "error",
                        "text": f"llm error: {e}"
                    })

                move_on = "let's move on" in response.lower() or "lets move on" in response.lower()

                if move_on:
                    st.session_state.questions_covered += 1

                    if st.session_state.questions_covered >= st.session_state.max_questions:
                        st.session_state.session_complete = True
                        try:
                            with st.spinner("generating feedback report..."):
                                report = st.session_state.feedback_gen.generate(
                                    conversation_history=st.session_state.agent.history,
                                    domain=st.session_state.domain,
                                    difficulty=st.session_state.difficulty
                                )
                            st.session_state.feedback_report = report
                        except Exception as e:
                            st.session_state.feedback_report = "OVERALL SCORE: N/A\n\nSUMMARY:\nFeedback generation failed. Please review your conversation above."
                        st.session_state.screen = "feedback"
                        st.rerun()
                    else:
                        try:
                            next_q = st.session_state.retriever.get_question(
                                domain=st.session_state.domain,
                                difficulty=st.session_state.difficulty,
                                exclude_ids=st.session_state.agent.questions_asked
                            )
                            if next_q:
                                st.session_state.agent.questions_asked.append(next_q["id"])
                                st.session_state.agent.add_to_history("ai", next_q["question"])
                                st.session_state.conversation.append({
                                    "role": "interviewer",
                                    "text": next_q["question"]
                                })
                                st.session_state.tts.speak(next_q["question"])
                            else:
                                closing = "we've covered all available questions for this session."
                                st.session_state.conversation.append({
                                    "role": "interviewer",
                                    "text": closing
                                })
                                st.session_state.tts.speak(closing)
                                st.session_state.screen = "feedback"
                        except Exception as e:
                            st.session_state.conversation.append({
                                "role": "error",
                                "text": f"failed to fetch next question: {e}"
                            })
                else:
                    st.session_state.conversation.append({
                        "role": "interviewer",
                        "text": response
                    })
                    st.session_state.tts.speak(response)

                st.rerun()

            except Exception as e:
                st.error(f"unexpected error: {e}")

    with col2:
        if st.button("end session"):
            try:
                with st.spinner("generating feedback report..."):
                    report = st.session_state.feedback_gen.generate(
                        conversation_history=st.session_state.agent.history,
                        domain=st.session_state.domain,
                        difficulty=st.session_state.difficulty
                    )
                st.session_state.feedback_report = report
            except Exception as e:
                st.session_state.feedback_report = "OVERALL SCORE: N/A\n\nSUMMARY:\nFeedback generation failed. Please review your conversation above."
            st.session_state.screen = "feedback"
            st.rerun()


def render_feedback():
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <p style='color:#00ff41; font-size:11px; letter-spacing:0.2em;'>── INTERVIEW FEEDBACK REPORT ──</p>
    </div>
    """, unsafe_allow_html=True)

    report = st.session_state.feedback_report

    if not report:
        st.error("no feedback available.")
        if st.button("▶  start new session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        return

    lines = report.split("\n")
    score_line = next((l for l in lines if "OVERALL SCORE" in l), None)
    if score_line:
        score = score_line.replace("OVERALL SCORE:", "").strip()
        st.markdown(f"<div class='score-display'>{score}</div>", unsafe_allow_html=True)

    sections = {
        "TOPICS COVERED": [],
        "STRENGTHS": [],
        "GAPS IDENTIFIED": [],
        "IMPROVEMENT SUGGESTIONS": [],
        "SUMMARY": []
    }

    current_section = None
    for line in lines:
        line = line.strip()
        for section in sections:
            if line.startswith(section):
                current_section = section
                break
        else:
            if current_section and line and not line.startswith("OVERALL"):
                sections[current_section].append(line)

    for section, content in sections.items():
        if content:
            st.markdown(f"""
            <div class='feedback-section'>
                <p class='section-label'>// {section.lower()}</p>
                <p style='color:#c0c0c0; margin:0; line-height:1.8;'>{"<br>".join(content)}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶  start new session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


init_session()

if st.session_state.screen == "setup":
    render_setup()
elif st.session_state.screen == "interview":
    render_interview()
elif st.session_state.screen == "feedback":
    render_feedback()