# Extending our speaker recognition model for authentication

This repo contains all the work done till now during the UST Internship June-July 2020.

### UST_learning_resources.odt

This file contains links for all the important topics to be learnt for better understanding.

### Documentation_On_Learning_Journey_and_Key-concepts.pdf

This pdf file contains information about all the key-concepts learnt during the initial weeks of the Internship.


### You-Only-Speak_Once

[Original-Repo](https://github.com/sampathkethineedi/You-Only-Speak-Once/tree/master/fbank_net)

This is a slight modification of the above mentioned Original-Repo.
The recognise and the register functions have been modified. Resemblyzer has been used to obtain the embeddings instead of extracting through the model(FBankNet).
The threshold for similarity is kept as **0.82**. If the embedding's inner product is above the threshold, the speaker's voice is recognised.

*This threshold can be changed in [test_api.py](https://github.com/ust-ilabs/iitp-jun2020/blob/spk-recog/You-Only-Speak-Once/fbank_net/demo/test_api.py)*

#### Pre-requisites:
Make sure all those mentioned in the [requirements_demos.txt](https://github.com/ust-ilabs/iitp-jun2020/blob/spk-recog/You-Only-Speak-Once/fbank_net/demo/requirements_demos.txt) and [requirements_package.txt](https://github.com/ust-ilabs/iitp-jun2020/blob/spk-recog/You-Only-Speak-Once/fbank_net/demo/requirements_package.txt) have beem installed.
 
#### For Testing:

run `python test_api.py`

Functionality (Can be run on Postman)

`/GET localhost:5000/recognise`

`/POST localhost:5000/register {"speaker": < name >}`

`/POST localhost:5000/delete {"speaker": < name >}`

### Streamlit_Resemblyzer_Week4

This folder contains the frontend.py file for running the web app using streamlit.

### Installing

- Streamlit

```
$ pip install streamlit
```

### Running the web app 

- Running streamlit app

Run the following command in a terminal
*Make sure you are in the correct folder*
```
$ streamlit run frontend.py
```

- Running the flask app
```
$ python test_api.py
```

