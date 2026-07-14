# AI Voice Mock Interviewer

A voice-based mock interview system for college students preparing for placement interviews. Practice DSA, System Design (LLD), and HR interviews with an AI interviewer that listens to your spoken answers, asks targeted follow-up questions, and generates a detailed feedback report at the end.

---

## What it does

- Asks questions from a curated bank of 230+ tagged interview questions
- Listens to your spoken answers using Deepgram STT
- Evaluates your answer and asks intelligent follow-up questions using Groq LLM
- Speaks responses out loud using pyttsx3 TTS
- Generates a structured feedback report with score, strengths, gaps, and suggestions
- Clean terminal-style web UI built with Streamlit

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (Llama 3.3 70B) |
| Speech to Text | Deepgram Nova-2 |
| Text to Speech | pyttsx3 (offline) |
| Vector Database | ChromaDB + sentence-transformers |
| Frontend | Streamlit |
| Language | Python 3.13 |

---

## Project Structure

```
voice-mock-interviewer/
├── backend/
│   ├── app/
│   │   ├── orchestrator.py    
│   │   ├── audio_handler.py   
│   │   ├── tts_handler.py     
│   │   └── feedback.py         
│   └── rag/
│       ├── ingest.py         
│       ├── retriever.py      
│       └── questions/
│           ├── dsa.json
│           ├── system_design.json
│           └── hr.json
├── frontend/
│   └── main.py                
├── chroma_db/                  
├── .env                       
├── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Devanshuk29/voice-mock-interviewer.git
cd voice-mock-interviewer
```

### 2. Create and activate virtual environment

```bash
python -m venv venv --without-pip
venv\Scripts\activate
python -m ensurepip --upgrade
```

### 3. Install dependencies

```bash
pip install python-dotenv langchain langchain-groq chromadb sentence-transformers deepgram-sdk==3.10.1 pyaudio pyttsx3 streamlit pygame
```

### 4. Set up API keys

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_key
DEEPGRAM_API_KEY=your_deepgram_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id
CHROMA_PERSIST_DIR=./chroma_db
```

Get your keys from:
- Groq: [console.groq.com](https://console.groq.com) — free
- Deepgram: [console.deepgram.com](https://console.deepgram.com) — free $200 credit

### 5. Build the question database

```bash
python backend/rag/ingest.py
```

### 6. Run the app

```bash
streamlit run frontend/main.py
```

---

## How to use

1. Select a domain — DSA, System Design, or HR
2. Select difficulty — Easy, Medium, or Hard
3. Click **initialize session**
4. When the interviewer asks a question, click **record answer**
5. Speak your answer and pause for 5 seconds when done
6. The interviewer will probe your answer with follow-up questions
7. After 5 questions, a detailed feedback report is generated

---
