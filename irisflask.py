import numpy as np
import pickle
from flask import Flask, jsonify, request, render_template

#create a flask app
app = Flask(__name__)

#load the pickled model
model = pickle.load(open("iris.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("iris.html")

@app.route("/predict", methods=["POST"])
def predict():
    input = [float (x) for x in request.form.values()]
    input_features = [np.array(input)]
    prediction = model.predict(input_features)
    
    return render_template("iris.html", prediction_text = "Your flower species is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)

