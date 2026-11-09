import os
import logging
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from chatbot_engine import OptimizedTurkishChatbotEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- Chatbot Engine Initialization ---
# Ensure knowledge_base.md exists and is configured correctly
KNOWLEDGE_BASE_PATH = "data/knowledge_base.md"

try:
    chatbot_engine = OptimizedTurkishChatbotEngine(
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        model_name="gemma2:2b",  # Or your preferred Ollama model
        embedding_model="all-minilm"  # Or your preferred Ollama embedding model
    )
    logger.info("Chatbot engine successfully initialized and ready to serve requests.")
except Exception as e:
    logger.error(f"Failed to initialize the chatbot engine: {e}")
    chatbot_engine = None  # If initialization fails, set to None to handle later


# --- Webhook Endpoint for Twilio ---
@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    """
    This endpoint handles incoming WhatsApp messages from Twilio.
    """
    if chatbot_engine is None:
        logger.error("Request received but chatbot engine is not available.")
        return "Chatbot not ready.", 503

    incoming_msg = request.values.get('Body', '').strip()
    sender_number = request.values.get('From', '')

    logger.info(f"Incoming message from {sender_number}: '{incoming_msg}'")

    # Create a Twilio MessagingResponse object
    resp = MessagingResponse()

    if not incoming_msg:
        logger.warning("Received an empty message.")
        resp.message("Üzgünüm, boş mesajlar yanıtlanamıyor.")
        return str(resp)

    # Get the response from your chatbot engine
    try:
        response_data = chatbot_engine.get_response(incoming_msg)
        bot_response = response_data["answer"]
        logger.info(f"Generated response: '{bot_response}'")
    except Exception as e:
        logger.error(f"Error during chatbot response generation: {e}")
        bot_response = "Üzgünüm, şu anda bir sorunla karşılaşıldı. Lütfen daha sonra tekrar deneyin."

    # Use the MessagingResponse object to send the reply
    resp.message(bot_response)

    return str(resp)


# --- Main Entry Point ---
if __name__ == "__main__":
    # To expose your localhost to the internet, you can use ngrok.
    # 1. Start your Flask app: flask run --port 5000
    # 2. In a new terminal, start ngrok: ngrok http 5000
    # 3. Copy the ngrok URL and paste it into your Twilio Sandbox settings.
    app.run(host='0.0.0.0', port=5000, debug=True)