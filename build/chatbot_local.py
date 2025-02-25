import torch
from transformers import pipeline
import os

class LocalChatBot:
    def __init__(self):
        #print("Initializing TinyLlama chatbot... This might take a few minutes on first run as it downloads the model.")
        try:
            self.pipe = pipeline(
                "text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("Model loaded successfully!")
            
            # Initialize system prompt
            self.messages = [
                {
                    "role": "system",
                    "content": "You are a helpful and friendly assistant."
                }
            ]
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def get_response(self, user_input: str) -> str:
        try:
            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})
            
            # Format messages using chat template
            prompt = self.pipe.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            outputs = self.pipe(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
            response = outputs[0]["generated_text"]
            
            # Extract only the new generated text
            response = response.replace(prompt, "").strip()
            
            # Add assistant's response to history
            self.messages.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def run(self):
        print("\nChatbot is ready! Type 'quit' to exit.")
        print("Note: First response might take a little longer to generate.")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
                
            response = self.get_response(user_input)
            print("\nBot:", response)

def main():
    # Check if XPU (GPU support) is available
    if torch.xpu.is_available():
        print("GPU detected! Using:", torch.xpu.get_device_name(0))
    else:
        print("No GPU detected. Running on CPU (this will be slower)")
    
    try:
        chatbot = LocalChatBot()
        chatbot.run()
    except Exception as e:
        print(f"Failed to initialize chatbot: {str(e)}")

if __name__ == "__main__":
    main()
