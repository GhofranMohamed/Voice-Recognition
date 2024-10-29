from flask import Flask, render_template, request, redirect, send_from_directory
import speech_recognition as sr
import joblib
import features
import os
import json
from spectrogram import *

global signalPath

app = Flask(__name__)

# AUDIO_FOLDER = 'D:/open the door/audio'
AUDIO_FOLDER = '.\\files'
IMG_FOLDER = '.\\static\\assets\images'


app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['IMG_FOLDER'] = IMG_FOLDER

ml_model = joblib.load('./rec_person.joblib')
ml_model_word = joblib.load('./tree_class_word.joblib')


# @app.route("/", methods=["GET", "POST"])
# def index():

#     return render_template('index.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    person = ""
    word = ""
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return {"error": "file not found"}, 404

        file = request.files["file"]

        if file:
            print(file)
            signalPath = os.path.join(AUDIO_FOLDER, file.filename)
            file.save(signalPath)
            df1 = features.extractWavFeatures(signalPath).split()
            person = ml_model.predict([df1])[0]
            df2 = features.extractWordFeatures(signalPath).split()
            word = ml_model_word.predict([df2])[0]
            plotGraph(person)
            svmImg = os.path.join(
                IMG_FOLDER, 'spectrograms'+str(variables.counter)+'.jpg')
            variables.counter += 1
            try:
                recognizer = sr.Recognizer()
                audioFile = sr.AudioFile(signalPath)
                with audioFile as source:
                    data = recognizer.record(source)
                transcript = recognizer.recognize_google(data, key=None)
                print(transcript)
            except sr.UnknownValueError:
                print("Could not understand")

    return {"person": str(person), "transcript": transcript, "word":  str(word), "svmImg": svmImg}, 200


@app.route("/", methods=["GET", "POST"])
def index():

    # plotGraph()
    # svmImg = os.path.join(IMG_FOLDER, 'spectrograms.jpg')

    return render_template('index.html')


@app.route('/spectrogram', methods=['GET'])
def spectrogram():

    if request.method == 'GET':
        plotGraph()
        return send_from_directory(directory=IMG_FOLDER, path='spectrograms.jpg')


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
