from flask import Flask, render_template, request, jsonify
from model import *

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("question:",input)
    return get_Chat_response(input)

# load model and tokenizer
model, tokenizer = get_model_and_tokenizer()

def get_Chat_response(text):
    return z_stream(text,model,tokenizer)

#############Test Streaming####################
# import random
# import time

# def get_Chat_response(text):
#     response = random.choice(
#         [
#             "Hello there! How can I assist you today?",
#             "Hi, human! Is there anything I can help you with?",
#             "Do you need help?",
#         ]
#     )
#     for word in response.split():
#         yield word + " "
#         time.sleep(1)
#######################################
        
if __name__ == '__main__':
    app.run(host='10.0.0.2', port=8502)