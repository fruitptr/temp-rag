import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env

def initialize_chatbot(filename):
    load_dotenv()

    # Define the base directory and persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

    # Define the filename to search for
      # Replace this with the actual filename from the request
    file_specific_directory = os.path.join(db_dir, filename.replace(".pdf", ""))

    # Define the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the specific vector store for the given file
    if os.path.exists(file_specific_directory):
        print(f"Loading vector store for {filename}...")
        db = Chroma(persist_directory=file_specific_directory, embedding_function=embeddings)
    else:
        raise FileNotFoundError(f"The vector store for {filename} does not exist.")

    # Create a retriever for querying the vector store
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # Create a ChatOpenAI model
    llm = ChatOpenAI(model="gpt-4o")

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question prompt
    qa_system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context to answer the question. Don't use information other than the retrieved context. If you don't know the answer, just say that I'm 1 month old right now, as I grow older, I will learn more and more. Can you please keep the questions easy? Keep the answer concise."
        "{context}"


        "Give response like this: "
        "===Normal Answer==="
        "[Provide the normal answer here.]"

        "===Explain to a 5-year-old==="
        "[Explain the answer in simple terms here.]"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain, retriever

def process_query(query, chat_history, rag_chain):
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result["answer"]

# Function to simulate a continual chat
def continual_chat(query, filename):
    rag_chain, retriever = initialize_chatbot(filename)
    chat_history = []  # Collect chat history here (a sequence of messages)
    # Process the user's query through the retrieval chain
    source = retriever.invoke(query)
    result = process_query(query, chat_history, rag_chain)
    # Display the AI's response
    print(f"TBD AI: {source[0].metadata}\n {result}")
    # Update the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result))
    print(source[0].page_content)
    # payload = [result, source[0]]
    # return payload

    # response_string = "===Normal Answer===\nBitcoin is a peer-to-peer electronic cash system that allows online payments to be sent directly from one person to another without using a financial institution. It uses cryptographic proof to solve the double-spending problem, ensuring that the same digital coin is not used more than once by maintaining an ongoing chain of transactions.\n\n===Explain to a 5-year-old===\nBitcoin is like internet money that you can send directly to someone else without needing a bank. It's safe because it uses special computer tricks to make sure no one can spend the same coin twice."

    # Splitting the response string into parts
    parts = result.split("===Explain to a 5-year-old===")

    # Extracting the normal answer
    normal_answer = parts[0].replace("===Normal Answer===", "").strip()

    # Extracting the explanation for a 5-year-old
    explanation_for_5_year_old = parts[1].strip()

    print("Normal Answer:\n", normal_answer)
    print("\nExplain to a 5-year-old:\n", explanation_for_5_year_old)

    return normal_answer, source[0], explanation_for_5_year_old
