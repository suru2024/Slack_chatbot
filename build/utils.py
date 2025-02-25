import yaml

# def load_tokens():
#     with open("token.yml", "r") as file:
#         data = yaml.safe_load(file)
#         return data.get("TOKEN", {}), data.get("RECIPIENT", {})
    

def load_tokens():
    """Loads tokens from token.yml and returns them as a dictionary."""
    try:
        with open("token.yml", "r") as file:
            data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise ValueError("Invalid token.yml format.")
            return data
    except Exception as e:
        raise ValueError(f"Error loading token.yml: {e}")

#         # return data.get("SLACK_BOT_TOKEN"), data.get("SLACK_APP_TOKEN")
#         # return data
        # return yaml.safe_load(file)

# import yaml

# def load_tokens():
#     with open("token.yml", "r") as file:
#         data = yaml.safe_load(file)

#     return {
#         "SLACK_BOT_TOKEN": data.get("TOKEN", {}).get("SLACK_BOT_TOKEN"),
#         "SLACK_APP_TOKEN": data.get("TOKEN", {}).get("SLACK_APP_TOKEN"),
#         "GEMINI_API_KEY": data.get("GEMINI_KEY", {}).get("GEMINI_API_KEY")
#     }

