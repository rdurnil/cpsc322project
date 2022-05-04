import os
import pickle
from flask import Flask, jsonify, request
from mysklearn.myclassifiers import MyKNeighborsClassifier

app = Flask(__name__)
@app.route("/", methods=["GET"])
def index():
    return "<h1>Welcome to the Ozone Level Predictor for Mount Rainier!!</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    temp = request.args.get("temp", "")
    humidity = request.args.get("humidity", "")
    radiation = request.args.get("radiation", "")
    precip = request.args.get("precip", "")
    windspeed = request.args.get("windspeed", "")
    winddir = request.args.get("winddir", "")
    shelter = request.args.get("shelter", "")

    prediction = predict_ozone([float(temp), float(humidity), float(radiation), float(precip), float(windspeed), float(winddir), float(shelter)])
    if prediction is not None:
        result = {"Predicted Ozone Level": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def predict_ozone(instance):
    infile = open("trained_knn.p", "rb")
    knn_classifier = pickle.load(infile)
    infile.close()

    print("neighbors:", knn_classifier.n_neighbors)
    try:
        prediction = knn_classifier.predict([instance])
        return prediction
    except:
        print("error")
        return None

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, port=port, host="0.0.0.0") # TODO: turn off debug