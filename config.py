# Import the module necessary to create parameters for generative AI
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Model parameters
PARAMETERS = {
    GenParams.DECODING_METHOD: "greedy", 
    GenParams.MAX_NEW_TOKENS: 256, 
}

# WatsonX credentials
# NOTE: Normally we'd need an API key 
    # But not in this case due to IBM Skill's Network Cloud IDE
CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com", 
    "project_id": "skills_network"
}

# Model IDs
LLAMA_MODEL_ID = "meta-llama/llama-3-2-11b-vision-instruct"
GRANITE_MODEL_ID = "ibm/granite-3-3-8b-instruct"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"