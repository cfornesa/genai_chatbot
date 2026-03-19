# Required to: 
    # authenticate
    # interact with API
    # define models
    # set parameters
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# Setup Credentials object to authenticate with IBM Watsonx AI where API key would normally be added
    # Instance of APIClient is created to allow for interaction with API
credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                   # api_key = "<YOUR_API_KEY>" # Normally you'd put an API key here, but we've got you covered here
                  )

# Defines the key settings for how the LLM generates its output
    # Adjust for the decoding method
        # Controls how the LLM selects its next token
        # Greedy by default
        # Sampling option to influence randomness of model choices (using temperature)
        # Greedy yields more deterministic responses
    # Adjust maximum new tokens that can be generated per response
        # Input and output tokens contribute to cost of model use
        # Import to manage usage
params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
	GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

# Initialize the model with the defined parameters + credentials
model = ModelInference(
    # model_id='ibm/granite-3-3-8b-instruct',
    model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8', 
    # model_id='mistralai/mistral-small-3-1-24b-instruct-2503', 
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

# Setup a text prompt
text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant who provides concise and accurate answers.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
What is the capital of Canada?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

# Use the generate method of model to get a response
model_response = model.generate(text)['results'][0]['generated_text']
print("Llama:")
print(model_response)
print()

mistral_text = """
system
[INST]You are an expert assistant who provides concise and accurate answers.[INST]

user
What is the capital of Canada?

assistant
"""

# Initialize the model with the defined parameters + credentials
mistral_model = ModelInference(
    model_id='mistralai/mistral-small-3-1-24b-instruct-2503', 
    params=params,
    credentials=credentials,
    project_id="skills-network"
)

# Use the generate method of model to get a response
mistral_response = mistral_model.generate(mistral_text)['results'][0]['generated_text']
print("Mistral:")
print(mistral_response)