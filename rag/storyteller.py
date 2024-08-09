import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def storyteller(characterList, plot, categoryList, languageCode, isRandom):

    # Load environment variables from .env
    load_dotenv()

    example = """
    {
        title: string,
        story: "Story should be in a single line. Use \n for new lines.",
        moralOfTheStory: string,
        languageCode: string,
    }

    """

    # Combine the query and the relevant document contents into the quiz generation prompt
    if isRandom == False:
        combined_input = (
            """
            You are the best story teller in the world. Your task is to generate a story by using the information provided to you. 
            All the information about the charaters is right here: {characterList}, the plot of the story will be {plot}, the 
            category list is as follows {categoryList}. 

            Note: Write the story in this language: {languageCode} and don't include ``` json ``` in the output

            You must output the result in a JSON format. An example output is provided below. Don't return anything other than the JSON.\n\n"
            {example}
            """
        ).format(
            characterList=characterList,
            plot=plot,
            categoryList=categoryList,
            example=example,
            languageCode=languageCode,
        )
    else:
        combined_input = """
            You are the best story teller in the world. Your task is to generate an interesting story.

            Note: Write the story in this language: {languageCode} and don't include ``` json ``` in the output

            You must output the result in a JSON format. An example output is provided below. Don't return anything other than the JSON.

            
            {example}
            """.format(
            example=example,
            languageCode=languageCode,
        )

    # Create a ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")

    # Define the messages for the model
    messages = [
        SystemMessage(content="You are the best story teller."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)

    # Display the full result and content only
    # print("\n--- Generated Response ---")
    print(result.content)

    try:
        storyteller_json = json.loads(result.content)  # Parse the JSON content
        return storyteller_json  # Return the parsed JSON object
    except json.JSONDecodeError:
        raise ValueError("The generated content is not valid JSON.")
