from __future__ import print_function

# This is a placeholder for a Google-internal import.
from flask import Flask
from flask_cors import CORS, cross_origin

from sklearn.externals import joblib

clf = joblib.load('/home/beaumaris/echo/experiment/sound_classifier.pkl')

import numpy as np
import librosa

from flask import jsonify

app = Flask(__name__)
CORS(app)

@app.route('/<file_name>')
def main(file_name):
      
      # data = '/home/brianphiri/Documents/sound_classification_take_one/serving/tensorflow_serving/example/audio/fold1/24074-1-0-2.wav'
      # data = '/home/beaumaris/echo/experiment/sounds/'+file_name
      data = '/home/beaumaris/echo/experiment/1505639950135.wav'
      print("file : "+data)
      sound = parse_audio(data)
      result = clf.predict(sound)
      # print("sound : %s", sound)
      # return jsonify({'prediction': list(result)})
      print("result : "+str(list(result)))
      return str(list(result))


def featureExtraction(file):
      print('Extracting', file)
      y, sr = librosa.load(file)
      stft = np.abs(librosa.stft(y))
      chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
      mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
      mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T,axis=0)
      contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
      tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr).T,axis=0)
      return chroma, mfccs, mel, contrast, tonnetz
  
def parse_audio(data):
      print("parsing file")
      features, labels = np.empty((0,193)), np.empty(0)
      try:
        chroma, mfccs, mel, contrast, tonnetz = featureExtraction(data)
        extracted_features  = np.hstack([chroma, mfccs, mel, contrast, tonnetz])
        features = np.vstack([features,extracted_features])
        return np.array(features)
      except Exception as ex:
        print("Could not parse file")

if __name__ == "__main__":
    app.run()
