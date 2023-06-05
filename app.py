from flask import Flask, render_template, request
import joblib

from pathlib import Path


app = Flask(__name__)
# https://stackoverflow.com/a/75262400/3693763
# https://stackoverflow.com/a/69635854/3693763
app.config.update(TEMPLATES_AUTO_RELOAD=True)



path = Path(__file__).parent
file_name = "/model.pkl"
model = joblib.load(
    open(f"{path}{file_name}", "rb")
)


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_music_genre():
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    prediction = model.predict([[age, gender]])
    return render_template("index.html", age=age, gender=gender, prediction=prediction)


if __name__ == "__main__":
    app.run()
