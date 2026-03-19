# Interface to interact with IBM Watsonx AI models
from langchain_ibm import ChatWatsonx
# Allows creation of dynamic prompts with placeholders for AI input
    # Change from langchain.prompts due to restructuring
from langchain_core.prompts import PromptTemplate
# Predefined configuration values from config.py
from config import PARAMETERS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID
# BaseModel and Field to define JSON output structure
from pydantic import BaseModel, Field
# Use JsonOutputParser to automatically parse and validate the AI output
from langchain_core.output_parsers import JsonOutputParser


# Define a class to properly output JSON structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message.")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive).")
    response: str = Field(description="Suggested response to the user.")
    category: str = Field(description="Category of the inquiry (e.g., billing, technical, general).")
    action: str = Field(description="Recommended action for the support representative.")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object = AIResponse)


# Function to initialize a model
def initialize_model(model_id):
    return ChatWatsonx(
        model_id = model_id, 
        url = "https://us-south.ml.cloud.ibm.com", 
        project_id = "skills-network", 
        params = PARAMETERS
    )

# Initialize models
llama_llm = initialize_model(LLAMA_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
mistral_llm = initialize_model(MISTRAL_MODEL_ID)

# Prompt templates instantiated using PromptTemplate class
    # Variables are reused with different content
llama_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}
    {format_instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "format_instructions", "user_prompt"]
)

granite_template = PromptTemplate(
    template='''System: 
    {system_prompt}
    {format_instructions}
    
    User: {user_prompt}
    
    Assistant: 
    ''',
    input_variables=["system_prompt", "format_instructions", "user_prompt"]
)

mistral_template = PromptTemplate(
    template='''<s>[INST]{system_prompt}
    {format_instructions}
    {user_prompt}[/INST]''',
    input_variables=["system_prompt", "format_instructions", "user_prompt"]
)


# Chain a prompt template and LLM together using | to create a sequence
    # Handles the process of formatting prompts and getting responses
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser  # Add json_parser to chain
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'format_instructions': json_parser.get_format_instructions()  # Pass format instructions at invoke time
    })
    

# Model-specific response functions to customize get_ai_response outputs
    # Modular approach which makes it simpler to add new models or modify existing models
def llama_response(system_prompt, user_prompt): 
    return get_ai_response(llama_llm, llama_template, system_prompt, user_prompt)

def granite_response(system_prompt, user_prompt): 
    return get_ai_response(granite_llm, granite_template, system_prompt, user_prompt)

def mistral_response(system_prompt, user_prompt): 
    return get_ai_response(mistral_llm, mistral_template, system_prompt, user_prompt)