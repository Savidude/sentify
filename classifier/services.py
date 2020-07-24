from flask import Flask
from flask_cors import CORS

from flask import request
from flask import jsonify

from classify_manager import predict_many
from model.classify_request import ClassifyRequest

app = Flask(__name__)
CORS(app)


@app.route('/classify', methods=['POST'])
def classify():
    req_data = request.get_json()
    classify_request = ClassifyRequest(req_data['tweets'])
    predictions = predict_many(classify_request)
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(threaded=False)
