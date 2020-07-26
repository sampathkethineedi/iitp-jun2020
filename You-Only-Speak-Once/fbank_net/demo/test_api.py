import numpy as np
from flask import Flask, request, Response
import pyaudio
from preprocessing import extract_fbanks
from predictions import get_embeddings, get_cosine_distance
import wave
import pickle
import os
import glob

from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from tqdm import tqdm

app = Flask(__name__)
p = pyaudio.PyAudio()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

DATA_DIR = 'data_files/'
THRESHOLD = 0.45


def load_spk():
    try:
        with open('speaker.pkl', 'rb') as f:
            speakers = pickle.load(f)
            print('Loaded embeddigs of speakers: ', speakers)
            f.close()
    except FileNotFoundError:
        speakers = []
    return speakers


def save_spk(speakers):
    with open('speaker.pkl', 'wb') as f:
        pickle.dump(speakers, f)
        f.close()


@app.route('/delete', methods=['POST'])
def delete():
    speakers = load_spk()
    r = request.get_json()

    if r['speaker'] == 'ALL':
        del_speakers = speakers
    else:
        del_speakers = r['speaker']

    for spk in del_speakers:
        files = glob.glob(DATA_DIR+spk+'/*')
        for f in files:
            os.remove(f)
        os.rmdir(DATA_DIR+spk)
        speakers.remove(spk)

    save_spk(speakers)
    res = 'DELETED: '+str(del_speakers)
    return res


@app.route('/register', methods=['POST'])

def register():
    speakers = load_spk()
    r = request.get_json()
    print("* LISTENING to speaker: " + r['speaker'])

    filename = listen('register', r, seconds=15)
    speakers.append(r['speaker'])

    save_spk(speakers)
    wav = preprocess_wav(filename)


    encoder = VoiceEncoder()
    embeddings = encoder.embed_utterance(wav)
    np.set_printoptions(precision=3, suppress=True)
    print('shape of embeddings: {}'.format(embeddings.shape), flush=True)
    np.save(DATA_DIR + r['speaker'] + '/embeddings.npy', embeddings)
    return Response('SUCCESS', mimetype='application/json')

@app.route('/recognise', methods=['GET'])


def recognise():
    speakers = load_spk()
    #positives_mean_list = []
    similarity_dict = {}
    r = request.get_json()
    filename = listen('recognise', r, seconds=10)
    wav = preprocess_wav(filename)

    encoder = VoiceEncoder()
    embeddings = encoder.embed_utterance(wav)
    print("Shape of embeddings: %s" % str(embeddings.shape))

    for speaker in speakers:
        stored_embeddings = np.load(DATA_DIR + speaker + '/embeddings.npy')
        #print('---', speaker, '---')
        utt_sim_matrix = np.inner(embeddings, stored_embeddings)
        #print(utt_sim_matrix)
        similarity_dict.update({speaker: utt_sim_matrix})
        
    print(similarity_dict)
    key_max = max(similarity_dict.keys(), key=(lambda k: similarity_dict[k]))
    #print(key_max)
    
    if (similarity_dict[key_max] >= 0.82):
    	return Response('///MATCH: '+key_max+' ///', mimetype='application/json')
    else: 
    	return Response('!!! NO MATCH !!!', mimetype='application/json')


def listen(act_type, r, seconds=10):
    speakers = load_spk()
    frames = []

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        if act_type == 'register':
            speaker = r['speaker']
            print("* LISTENING to speaker: " + r['speaker'], len(frames))
        else:
            speaker = 'UNK'
            print("* LISTENING ... ", len(frames))

        if len(frames) == int(RATE / CHUNK * seconds):
            break

    dir_ = DATA_DIR + speaker
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    filename = DATA_DIR + speaker + '/sample.wav'
    wf = wave.open(filename, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename


if __name__ == '__main__':
    app.run(host='0.0.0.0')

