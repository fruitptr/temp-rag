import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json

def link_chatbot(text, query):
    load_dotenv()

    transcript = text
    
    prompt = ("""You are a helpful assistant. Use the following pieces of retrieved transcript from a YouTube video to answer the question. 
              If the answer is not directly found in the retrieved transcript, provide a relevant interpretation or context based on common knowledge, but make it clear that it's not directly from the transcript. 
              If the question is not relevant to the video, please say that "I could not find a relevant answer in the video. Can you rephrase the question please?".

            The transcript from the video is as follows:
            {transcript}

            Give response in the following format. I need to parse the response to extract the answer and explanation for a 5-year-old so please follow the format strictly:
            "===Normal Answer==="
            [Provide the answer here, using information from the transcript or relevant context.]

            ===Explain to a 5-year-old===
            [Explain the answer in simple terms here.]
            """).format(transcript=transcript)
    

    print("Prompt:", prompt)
    
    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)
    result = result.content

    print("Result:", result)
    
    if result == "I could not find a relevant answer in the video. Can you rephrase the question please?":
        normal_answer = "I could not find a relevant answer in the video. Can you rephrase the question please?"
        explanation_for_5_year_old = "I could not find a relevant answer in the video. Can you rephrase the question please?"
        return normal_answer, explanation_for_5_year_old
    
    parts = result.split("===Explain to a 5-year-old===")

    # Extracting the normal answer
    normal_answer = parts[0].replace("===Normal Answer===", "").strip()

    # Extracting the explanation for a 5-year-old
    explanation_for_5_year_old = parts[1].strip()

    print("Normal Answer:\n", normal_answer)
    print("\nExplain to a 5-year-old:\n", explanation_for_5_year_old)

    return normal_answer, explanation_for_5_year_old