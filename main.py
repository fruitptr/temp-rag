from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.quiz_generator import one_off
from rag.chatbot import continual_chat
from rag.transcribe_video import transcribe_video
from rag.link_chatbot import link_chatbot
from rag.storyteller import storyteller
import os
from typing import Optional
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

chat_histories = {}

class QuizRequest(BaseModel):
    filename: Optional[str] = None
    text: Optional[str] = None
    difficulty: str
    noOfQuestions: int
    quizType: str

@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    try:
        result = one_off(request.filename, request.text, request.difficulty, request.noOfQuestions, request.quizType)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    query: str
    filename: Optional[str] = None
    text: Optional[str] = None
    session_id: Optional[str] = None

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        if request.filename:
            session_id = request.session_id
            # Retrieve existing chat history or start a new one
            chat_history = chat_histories.get(session_id, [])

            # Process the query with the current chat history
            normal_answer, source, explanation_for_5_year_old, updated_chat_history = (
                continual_chat(request.query, request.filename, chat_history)
            )

            # Update the global chat history
            chat_histories[session_id] = updated_chat_history

            return {
                "response": normal_answer,
                "source": source.metadata,
                "explanation_for_5_year_old": explanation_for_5_year_old,
                "chat_history": updated_chat_history
            }
        else:
            normal_answer, explanation_for_5_year_old = link_chatbot(request.text, request.query)
            return {
                "response": normal_answer,
                "explanation_for_5_year_old": explanation_for_5_year_old
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EvaluateRequest(BaseModel):
    filename: str
    originalQuizJson: dict
    userAnswerQuizJson: dict

def get_context_from_chroma(filename, originalQuizJson):
    # Implement the function to get context from Chroma
    load_dotenv()

    # Define the base directory and persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "rag", "db", "chroma_db_with_metadata")

    # Define the filename to search for
    # filename = "story.pdf"  # Replace with the actual filename from the request
    file_specific_directory = os.path.join(db_dir, filename.replace(".pdf", ""))
    print(file_specific_directory)

    # Define the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the specific vector store for the given file
    if os.path.exists(file_specific_directory):
        print("File exists")
        print(f"Loading vector store for {filename}...")
        db = Chroma(persist_directory=file_specific_directory, embedding_function=embeddings)
    else:
        raise FileNotFoundError(f"The vector store for {filename} does not exist.")

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    query = " ".join([str(value) for key, value in originalQuizJson.items()])
    relevant_docs = retriever.invoke(query)

    return relevant_docs


def query_llm(prompt):
    # Implement the function to query the LLM
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a evaluator."),
        HumanMessage(content=prompt),
    ]

    # Invoke the model with the combined input
    print("BEFORE INVOKING")
    result = model.invoke(messages)
    print("AFTER INVOKING")
    return result

@app.post('/evaluate')
def evaluate_quiz(request: EvaluateRequest):
    try:
        context = get_context_from_chroma(request.filename, request.originalQuizJson)
        
        prompt = f"""
        Context: {context}

        Evaluate the following answers based on the provided context. Be strict and only give the answer as correct if the essence of the answer is correct as per the question. Don't be fooled if the userAnswer contains many words. Don't return anything other than the JSON. The JSON input and desired JSON output format are shown below:

        Input JSON:
        {request.userAnswerQuizJson}

        Desired Output JSON format: 
        {{
            "1": {{
                "correctAnswer": "option1",
                "userAnswer": "option1",
                "isCorrect": true
            }},
            "2": {{
                "correctAnswer": "The war began when Nazi Germany invaded Poland in 1939 and raged across the globe until 1945",
                "userAnswer": "The war began when Nazi Germany invaded Poland in 1923 and raged across the globe until 1945",
                "isCorrect": false
            }}
        }}
        
        Only output the JSON, nothing else. Don't add any extra text before or after the JSON. 

        """

        result_json = query_llm(prompt)
        print("RESULT JSON CONTENT", result_json.content)
        # result_json = json.loads(result_json.content[0])
        return {"response": json.loads(result_json.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class VideoToTextRequest(BaseModel):
    youtubelink: str

@app.post("/video-to-text")
def video_to_text(request: VideoToTextRequest):
    try:
        response = transcribe_video(request.youtubelink)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/audio-to-text")
async def audio_to_text(file: UploadFile = File(...)):
    try:
        # Read the audio file
        audio_file = await file.read()
        audio_path = os.path.join(os.getcwd(), file.filename)

        # Save the audio file temporarily
        with open(audio_path, "wb") as f:
            f.write(audio_file)

        # Use OpenAI to transcribe the audio
        client = OpenAI()
        with open(audio_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio
            )
        
        # Delete the temporary audio file after processing
        os.remove(audio_path)
        
        return {"transcript": transcript.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class StoryRequest(BaseModel):
    characterList: list
    plot: str
    categoryList: list
    languageCode: str
    isRandom: bool


@app.post("/generate-story")
def generate_story(request: StoryRequest):
    try:
        result = storyteller(
            request.characterList,
            request.plot,
            request.categoryList,
            request.languageCode,
            request.isRandom,
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    porttoRun = os.getenv('PORT')
    uvicorn.run(app, host="0.0.0.0", port=porttoRun)