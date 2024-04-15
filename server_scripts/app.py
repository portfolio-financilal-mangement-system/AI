from flask import Flask, jsonify
from result import result
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    # Call your imported function and get the data
    data = result()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)