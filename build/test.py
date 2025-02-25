import requests
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from utils import load_tokens

# ✅ Load tokens
tokens = load_tokens()

# ✅ Retrieve API keys properly
bot_token = tokens.get("TOKEN", {}).get("SLACK_BOT_TOKEN")
app_token = tokens.get("TOKEN", {}).get("SLACK_APP_TOKEN")
gemini_api_key = tokens.get("GEMINI_KEY", {}).get("GEMINI_API_KEY")

# ✅ Check if keys are loaded correctly
if not gemini_api_key:
    raise ValueError("Gemini API key is missing. Check token.yml and ensure correct formatting.")
if not bot_token or not app_token:
    raise ValueError("Slack bot tokens are missing. Check token.yml.")

# ✅ Initialize Slack app
slack_app = App(token=bot_token)

class GeminiChatBot:
    def __init__(self):
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_api_key}"
        self.headers = {"Content-Type": "application/json"}
        self.conversations = {}

    def get_response(self, user_id: str, user_input: str) -> str:
        try:
            if user_id not in self.conversations:
                self.conversations[user_id] = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]

            self.conversations[user_id].append({"role": "user", "content": user_input})

            payload = {
                "contents": [{"parts": [{"text": user_input}]}]  # ✅ Correct payload format
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response_data = response.json()

            # ✅ Check for API errors
            if "error" in response_data:
                print(f"API Error: {response_data['error']}")
                return f"Error: {response_data['error'].get('message', 'Unknown error')}"

            # ✅ Extract response text (latest API format)
            response_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "I couldn't generate a response.")

            self.conversations[user_id].append({"role": "assistant", "content": response_text})

            return response_text
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return "I encountered an error while processing your request."

# ✅ Initialize chatbot instance
chatbot = GeminiChatBot()

@slack_app.event("message")
def handle_slack_message(event, say):
    user_id = event.get("user")
    message_text = event.get("text")
    if user_id and message_text:
        print(f"Received message from {user_id}: {message_text}")
        response_text = chatbot.get_response(user_id, message_text)
        say(f"<@{user_id}> {response_text}")

# ✅ Start the Slack bot
if bot_token and app_token:
    print("Starting Slack bot...")
    handler = SocketModeHandler(slack_app, app_token)
    handler.start()
else:
    print("Invalid Slack tokens.")
