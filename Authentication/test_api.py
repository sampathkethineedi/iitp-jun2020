import numpy as np
from flask import Flask, request, Response, jsonify
import pyaudio
import wave
import pickle
import os
import glob

from resemblyzer import preprocess_wav, VoiceEncoder
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

@app.route('/')
def welcome():
	return "Welcome!!!"

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
        for spk in del_speakers:
            files = glob.glob(DATA_DIR+spk+'/*')
            print("spk = ",spk)
            for f in files:
                os.remove(f)
            os.rmdir(DATA_DIR+spk)
            speakers.remove(spk)
    else:
        del_speakers = r['speaker']
        print("r[speaker] = ",r['speaker'])
        files = glob.glob(DATA_DIR+del_speakers+'/*')
        for f in files:
            os.remove(f)
        os.rmdir(DATA_DIR+del_speakers)
        speakers.remove(del_speakers)


    save_spk(speakers)
    res = 'DELETED: '+str(del_speakers)
    return jsonify({'del':res})


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
    similarity_dict = {}
    r = request.get_json()
    filename = listen('recognise', r, seconds=10)
    wav = preprocess_wav(filename)

    encoder = VoiceEncoder()
    embeddings = encoder.embed_utterance(wav)
    print("Shape of embeddings: %s" % str(embeddings.shape))

    for speaker in speakers:
        stored_embeddings = np.load(DATA_DIR + speaker + '/embeddings.npy')
        utt_sim_matrix = np.inner(embeddings, stored_embeddings)
        similarity_dict.update({speaker: utt_sim_matrix})
        
    print(similarity_dict)
    key_max = max(similarity_dict.keys(), key=(lambda k: similarity_dict[k]))
    
    if (similarity_dict[key_max] >= 0.82):
        mes = ['MATCH!',key_max]
        return(jsonify({"rec spk": mes}))
    else:
        mes = ['NO MATCH!']
        return(jsonify({"rec spk": mes}))

@app.route('/authenticate', methods=['POST'])
def authenticate():
    speakers = load_spk()
    similarity_dict = {}
    r = request.get_json()
    print("AUTHENTICATING: " + r['speaker'])
    
    spk_name = r['speaker']
    
    filename = listen('authenticate', r, seconds=10)
    wav = preprocess_wav(filename)
    
    encoder = VoiceEncoder()
    embeddings = encoder.embed_utterance(wav)
    print("Shape of embeddings: %s" % str(embeddings.shape))
    
    flag = 0
    for speaker in speakers:
        if(speaker == spk_name):
            flag = 1
            stored_embeddings = np.load(DATA_DIR + speaker + '/embeddings.npy')
            utt_sim_matrix = np.inner(embeddings, stored_embeddings)
            similarity_dict.update({speaker: utt_sim_matrix})
            print("SPEAKER FOUND...AUTHENTICATING...",speaker)
            break
    print(similarity_dict)
    if(flag == 0):
        print("SPEAKER NOT FOUND!!! Have you registered?")
        mes = ['SPEAKER NOT FOUND!!! Have you registered?']
        return(jsonify({"auth_spk": mes}))
    else:
        if(similarity_dict[spk_name]>=0.82):
            mes = ['MATCH!',spk_name]
            return(jsonify({"auth_spk": mes}))
        else:
            mes = ['NO MATCH!']
            return(jsonify({"auth_spk": mes}))
        

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
    app.run()


