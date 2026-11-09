import os
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv()


def test_twilio_connection():
    """Test basic Twilio connection and send message"""

    # Get credentials from .env file
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')

    if not account_sid or not auth_token:
        print("❌ Missing credentials in .env file")
        return False

    try:
        # Initialize Twilio client
        client = Client(account_sid, auth_token)

        # Test connection by getting account info
        account = client.api.accounts(account_sid).fetch()
        print(f"✅ Connected to Twilio account: {account.friendly_name}")

        # Send test message (update with YOUR phone number)
        your_phone = input("Enter your phone number with country code (e.g., +905551234567): ")

        message = client.messages.create(
            body='🤖 TechSoft WhatsApp Bot test message! Integration working perfectly.',
            from_='whatsapp:+14155238886',  # Twilio sandbox number
            to=f'whatsapp:{your_phone}'
        )

        print(f"✅ Test message sent! Message SID: {message.sid}")
        print("📱 Check your WhatsApp for the test message")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Twilio WhatsApp Integration...")
    test_twilio_connection()