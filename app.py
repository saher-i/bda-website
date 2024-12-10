from flask import Flask, jsonify, request
from process_data import process_ieee_cis_data
from train import main as train_model

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    """
    Fetch processed data for visualization.
    """
    try:
        edges = process_ieee_cis_data()  # Get processed data
        response_data = [{"relation": key, "count": len(value)} for key, value in edges.items()]
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    """
    Trigger model training and return status.
    """
    try:
        train_model()  # Trigger training logic
        return jsonify({"status": "success", "message": "Model trained successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

