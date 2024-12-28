import numpy as np
from utils.sequence_dataset import create_test_loader, create_test_dataset, get_sequences, inverse_scale
from flask import Flask, jsonify
from model.model import LSTMModel
import torch


app = Flask(__name__, static_url_path='/static')

look_back = 10
seq_counter = 0
test, test_clean = create_test_dataset(look_back)
test_dataset = create_test_loader(test, look_back)

model = LSTMModel(1, 5, 1)
state_dict = torch.load('./model/lstm_model.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    with torch.no_grad():
        predictions, actuals = [], []
        for sequences, targets in test_dataset:
            outputs = model(sequences)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        testPredict = inverse_scale(predictions.reshape(-1, 1))
        testY = inverse_scale(actuals.reshape(-1, 1))
        return jsonify({'sequences': get_sequences(test_clean, look_back), 'prediction': testPredict.tolist(), 'actual': testY.tolist()}), 200
    

if __name__ == '__main__':
    # Production
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
    # Development
    #app.run(host='0.0.0.0')