# WhatsApp assistant bot

A Turkish customer-service chatbot that answers on WhatsApp. Messages arrive via a Twilio webhook. The
bot finds the relevant sections of a company knowledge base with Ollama embeddings and writes the
reply with a small local LLM (Gemma 2, 2B). Nothing is sent to a hosted LLM API.

```
WhatsApp ──► Twilio ──► Flask /whatsapp ──► retrieve sections (all-minilm embeddings)
                                         └─► generate answer (gemma2:2b via Ollama) ──► reply
```

## Files

| | |
|---|---|
| `app.py` | Flask webhook for Twilio (`POST /whatsapp`) |
| `chatbot_engine.py` | knowledge-base loading, splitting, retrieval and answer generation |
| `data/knowledge_base.md` | example knowledge base for a fictional company, *TechSoft* |
| `test_twilio.py` | sends a test message with your Twilio credentials |

## Run

```bash
pip install -r requirements.txt
ollama pull gemma2:2b && ollama pull all-minilm
export TWILIO_ACCOUNT_SID=...  TWILIO_AUTH_TOKEN=...   # only for test_twilio.py
python app.py                                           # then point the Twilio WhatsApp sandbox webhook at /whatsapp
```

Credentials are read from environment variables and are never stored in the repository.
