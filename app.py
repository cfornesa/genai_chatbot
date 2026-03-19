# Import Flask, request, and jsonify modules from flask
from flask import Flask, request, jsonify, render_template
# Import the response functions fronm model.py
from model import llama_response, granite_response, mistral_response
# Import time
import time

# Instantiate Flask object
app = Flask(__name__)

# Generate index wrapper and function
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Generate route wrapper and function
    # Expected message and model fields
@app.route('/generate', methods=['POST'])
def generate():
    # This is where we'll add our AI logic later
    data = request.json
    user_message = data.get('message')
    model = data.get('model')

    # Error handling to ensure both message and model are present
    if not user_message or not model:
        return jsonify({"error": "Missing message or model selection"}), 400

    system_prompt = (
        "You are an AI assistant helping with customer inquiries. "
        "Analyze the customer message and provide: a summary of their message, "
        "a sentiment score from 0 (negative) to 100 (positive), "
        "a suggested response to the user, "
        "a category (billing, technical, or general), "
        "and a recommended action for the support representative."
    )

    start_time = time.time()

    # Error handling to ensure model exists
    try:
        if model == 'llama':
            result = llama_response(system_prompt, user_message)
        elif model == 'granite': 
            result = granite_response(system_prompt, user_message)
        elif model == 'mistral': 
            result = mistral_response(system_prompt, user_message)
        else:
            return jsonify({"error": "Invalid model seelction"}), 400

        result['duration'] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ensures Flask dev server runs when the file is executed
if __name__ == '__main__':
    app.run(debug = True)