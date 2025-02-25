from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch_directml
import torch
from transformers import pipeline
import time
import re

app = FastAPI(title="TinyLlama Chat API")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class LocalChatBot:
    def __init__(self):
        try:
            dml = torch_directml.device()
            self.pipe = pipeline(
                "text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16,
                device=dml
            )
            print("Model loaded successfully!")
            
            self.messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can provide both regular responses and code examples."
                }
            ]
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def is_code_request(self, user_input: str) -> bool:
        code_patterns = [
            r"(?i)write.*code",
            r"(?i)create.*program",
            r"(?i)generate.*code",
            r"(?i)implement.*function",
            r"(?i)code.*in (python|java|c\+\+|javascript)",
            r"(?i)program.*in (python|java|c\+\+|javascript)",
        ]
        return any(re.search(pattern, user_input) for pattern in code_patterns)

    def detect_language(self, text: str) -> str:
        languages = {
            "python": r"(?i)(python|\.py)",
            "javascript": r"(?i)(javascript|js|\.js)",
            "java": r"(?i)(java[^s]|\.java)",
            "c++": r"(?i)(c\+\+|cpp|\.cpp)",
            "html": r"(?i)(html|\.html)",
            "css": r"(?i)(css|\.css)"
        }
        
        for lang, pattern in languages.items():
            if re.search(pattern, text):
                return lang
        return "text"

    def get_response(self, user_input: str) -> dict:
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
            
            response_text = outputs[0]["generated_text"].replace(prompt, "").strip()

            # Structure the response
            if self.is_code_request(user_input):
                language = self.detect_language(response_text)
                response_data = {
                    "type": "code",
                    "content": {
                        "language": language,
                        "code": response_text,
                        "explanation": "Here's the requested code:"
                    }
                }
            else:
                response_data = {
                    "type": "text",
                    "content": {
                        "message": response_text
                    }
                }

            # Add assistant's response to history
            self.messages.append({"role": "assistant", "content": response_text})
            return response_data

        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return {
                "type": "error",
                "content": {
                    "message": f"Error generating response: {str(e)}"
                }
            }

    def get_conversation_history(self):
        return self.messages[1:]

    def clear_history(self):
        self.messages = [self.messages[0]]

chatbot = LocalChatBot()

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response_type: str
    content: dict
    generation_time: float

@app.post("/chat", response_model=ChatResponse)
async def chat_with_local_model(request: ChatRequest):
    try:
        start_time = time.time()
        response = chatbot.get_response(request.user_input)
        generation_time = time.time() - start_time

        return ChatResponse(
            response_type=response["type"],
            content=response["content"],
            generation_time=round(generation_time, 2)
        )

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"message": f"An error occurred: {str(e)}"}
        )

@app.get("/")
async def chat_page(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "title": "AI Chatbot",
            "chatbot_name": "TinyLlama Chat",
            "conversation_history": chatbot.get_conversation_history(),
            "dark_mode": True
        }
    )

@app.get("/chat-history")
async def get_chat_history():
    return JSONResponse(content=chatbot.get_conversation_history())

@app.post("/clear-history")
async def clear_history():
    chatbot.clear_history()
    return {"message": "Conversation history cleared"}

@app.post("/toggle-theme")
async def toggle_theme(dark_mode: bool = Form(...)):
    return {"dark_mode": dark_mode}

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"Unexpected error: {str(exc)}")  # For debugging
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred"}
    )

# Print GPU information on startup
if torch.cuda.is_available():
    print("GPU detected! Using:", torch.cuda.get_device_name(0))
elif torch_directml.is_available():
    print("Intel Iris Xe GPU is available!")
    print(torch_directml.device_name(0))
    #print(torch_directml.device_name(dml))
else:
    print("No GPU detected. Running on CPU (this will be slower)")
