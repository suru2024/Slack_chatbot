import requests
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from utils import load_tokens

# Load tokens
tokens, _ = load_tokens()
bot_token = tokens.get("SLACK_BOT_TOKEN")
app_token = tokens.get("SLACK_APP_TOKEN")
deepinfra_api_key = tokens.get("DEEPINFRA_API_KEY")

# Initialize Slack app
slack_app = App(token=bot_token)

class LocalChatBot:
    def __init__(self):
        self.api_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {deepinfra_api_key}",
            "Content-Type": "application/json"
        }
        self.conversations = {}

    def get_response(self, user_id: str, user_input: str) -> str:
        try:
            if user_id not in self.conversations:
                self.conversations[user_id] = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]

            self.conversations[user_id].append({"role": "user", "content": user_input})

            payload = {
                 "model": "google/gemma-2-27b-it",
                 "messages": self.conversations[user_id],
                 "temperature": 0.7,
                 "max_tokens": 256,
                 "top_p": 0.95
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response_data = response.json()

            # ✅ Check for API errors
            if "error" in response_data:
                print(f"API Error: {response_data['error']}")
                return "I encountered an API error while processing your request."

            # ✅ Ensure valid response format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                response_text = response_data["choices"][0]["message"]["content"].strip()
            else:
                print(f"Unexpected response format: {response_data}")
                return "I received an unexpected response format."

            self.conversations[user_id].append({"role": "assistant", "content": response_text})

            return response_text
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return "I encountered an error while processing your request."

# Initialize chatbot instance
chatbot = LocalChatBot()

@slack_app.event("message")
def handle_slack_message(event, say):
    user_id = event.get("user")
    message_text = event.get("text")
    if user_id and message_text:
        print(f"Received message from {user_id}: {message_text}")
        response_text = chatbot.get_response(user_id, message_text)
        say(f"<@{user_id}> {response_text}")

# Start the Slack bot
if bot_token and app_token:
    print("Starting Slack bot...")
    handler = SocketModeHandler(slack_app, app_token)
    handler.start()
else:
    print("Invalid Slack tokens.")
