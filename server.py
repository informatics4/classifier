import pika
import sys 

from sklearn.externals import joblib

import numpy as np
import librosa

# clf = joblib.load('sound_classifier.pkl')
clfr = joblib.load('sound_classifier_using_RandomForestClassifier.pkl')


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


# AMQP_URL = 'amqp://ehlbpomi:AODIFxJKO0QmTUqke2_FHjy5AKKcQ5ed@wasp.rmq.cloudamqp.com/ehlbpomi'
AMQP_URL = "localhost"
EXCHANGE_NAME = "audioFiles"

#connection = pika.BlockingConnection(pika.URLParameters(AMQP_URL))
connection = pika.BlockingConnection(pika.ConnectionParameters(host=AMQP_URL))

channel = connection.channel()

channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='fanout')

result = channel.queue_declare(exclusive=True)
queue_name = result.method.queue

channel.queue_bind(exchange=EXCHANGE_NAME, queue=queue_name)

print("waiting for files")

def callback(ch, method, properties, body):
    # print(" [x] %r" % body)
    file_path = '/home/brianphiri/code/java/recoder/'+body.decode('utf-8')
    sound = parse_audio(file_path)
    print("get result")
    result = clfr.predict(sound)

    channel.exchange_declare(exchange="audoClassification", exchange_type='fanout')

    message = str(list(result))
    channel.basic_publish(exchange='audoClassification', routing_key='', body=message)

    print("classification : "+str(list(result)))


channel.basic_consume(callback,queue=queue_name, no_ack=True)

channel.start_consuming()

