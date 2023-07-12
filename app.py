from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the machine learning model
model = joblib.load("model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input height from the form
    height = float(request.form["Enter height in cm's"])
    
    # Make a prediction using the loaded model
    weight = model.predict([[height]])
    
    # Round the weight to two decimal places
    weight = round(weight[0], 2)
    
    # Prepare the prediction text
    prediction_text = f"The predicted weight for a height of {height} cm is {weight} kg."
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)