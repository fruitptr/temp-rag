import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json

def one_off(filename, difficulty, noOfQuestions, quizType):
   
# Load environment variables from .env
    load_dotenv()

    # Define the base directory and persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

    # Define the filename to search for
    # filename = "story.pdf"  # Replace with the actual filename from the request
    file_specific_directory = os.path.join(db_dir, filename.replace(".pdf", ""))

    # Define the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the specific vector store for the given file
    if os.path.exists(file_specific_directory):
        print(f"Loading vector store for {filename}...")
        db = Chroma(persist_directory=file_specific_directory, embedding_function=embeddings)
    else:
        raise FileNotFoundError(f"The vector store for {filename} does not exist.")

    # Define the user's question parameters
    # difficulty = "hard"
    # noOfQuestions = 5
    # quizType = "FillInTheBlank"

    if quizType == 'MCQ':
        example = '''{
    "1": {
        "question": "Who is the president of the United States?",
        "options": {
        "option1": "Joe Biden",
        "option2": "Charlie Chaplin",
        "option3": "Katy Perry",
        "option4": "Peppa Pig"
        },
        "answer": "option1",
        "type": "mcq"
    },
    "2": {
        ...
    }
    }'''

    elif quizType == 'True/False':
        example = '''{
        "1": {
            "question": "The sky is blue.",
            "options": {
            "option1": "True",
            "option2": "False"
            },
            "answer": "option1",
            "type": "mcq"
        },
        "2": {
            "question": "2 + 2 equals 5.",
            "options": {
            "option1": "True",
            "option2": "False"
            },
            "answer": "option2",
            "type": "mcq"
        }
        }'''

    elif quizType == 'FillInTheBlank':
        example = '''{
        "1": {
            "question": "The capital of France is _____.",
            "answer": "Paris",
            "type": "blanks"
        },
        "2": {
            "question": "Water freezes at _____ degrees Celsius.",
            "answer": "0",
            "type": "blanks"
        }
        }'''

    elif quizType == 'Mixed':
        example = '''{
        "1": {
            "question": "Who is the president of the United States?",
            "options": {
            "option1": "Joe Biden",
            "option2": "Charlie Chaplin",
            "option3": "Katy Perry",
            "option4": "Peppa Pig"
            },
            "answer": "option1",
            "type": "mcq"
        },
        "2": {
            "question": "Why did World War 2 begin?",
            "answer": "The question is subjective. Please evaluate whether the answer is correct from the context above.",
            "type": "subjective"
        },
        "3": {
            "question": "Argentina were FIFA World Cup 2022 champions",
            "options": {
            "option1": "True",
            "option2": "False"
            },
            "answer": "option1",
            "type": "mcq"
        },
        "4": {
            "question": "The capital of France is _____.",
            "answer": "Paris",
            "type": "blanks"
        },
        "5": {
            ...
        }
        }
        
        Include a mix of MCQ, True/False, FillInTheBlank, and Subjective questions.'''

    elif quizType == 'Subjective':
        example = '''{
        "1": {
            "question": "Why did World War 2 begin?",
            "answer": "The question is subjective. Please evaluate whether the answer is correct from the context above.",
            "type": "subjective"
            },
        "2": {
            "question": "Briefly explain how the Electoral College works",
            "answer": "The question is subjective. Please evaluate whether the answer is correct from the context above.",
            "type": "subjective"
            },
        "3": {
            ...
            }
        }'''
    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    query = "Create a quiz from the story"  # The actual content isn't as important here
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        page_number = doc.metadata.get("page", "Unknown")  # Get the page number from metadata
        print(f"Document {i} (Page {page_number}):\n{doc.page_content}\n")

    # Combine the query and the relevant document contents into the quiz generation prompt
    combined_input = (
        "You are a knowledgeable assistant. Your task is to generate a quiz by using information only provided in the context below. "
        "Don't use information from external sources. Make sure the difficulty of the quiz is {difficulty} and there are a total of {noOfQuestions} questions. "
        "The type of quiz is {quizType}. The context is provided below:\n\n"
        "{context}\n\n"
        "You must output the result in a JSON format. An example output is provided below. Don't return anything other than the JSON.\n\n"
        "{example}"
    ).format(
        difficulty=difficulty,
        noOfQuestions=noOfQuestions,
        quizType=quizType,
        context="\n\n".join([f"Page {doc.metadata.get('page', 'Unknown')}:\n{doc.page_content}" for doc in relevant_docs]),
        example=example
    )

    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are a knowledgeable assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)

    # Display the full result and content only
    # print("\n--- Generated Response ---")
    print(result.content)

    try:
        quiz_json = json.loads(result.content)
        return quiz_json  # Return the parsed JSON object
    except json.JSONDecodeError:
        raise ValueError("The generated content is not valid JSON.")