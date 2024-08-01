from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.quiz_generator import one_off
from rag.chatbot import continual_chat
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QuizRequest(BaseModel):
    filename: str
    difficulty: str
    noOfQuestions: int
    quizType: str

@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    try:
        result = one_off(request.filename, request.difficulty, request.noOfQuestions, request.quizType)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    query: str
    filename: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        print(request.query)
        response = continual_chat(request.query, request.filename)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EvaluateRequest(BaseModel):
    fileName: str
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
    result = model.invoke(messages)
    return result

@app.post('/evaluate')
def evaluate_quiz(request: EvaluateRequest):
    try:
        context = get_context_from_chroma(request.fileName, request.originalQuizJson)
        
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
        }}"""

        result_json = query_llm(prompt)
        # result_json = json.loads(result_json.content[0])
        return {"response: ": json.loads(result_json.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
