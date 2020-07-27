# Extending our speaker recognition model for authentication

This repo contains all the work done till now during the UST Internship June-July 2020.

### UST_learning_resources.odt
This file contains links for all the important topics to be learnt for better understanding.

### Documentation_On_Learning_Journey_and_Key-concepts.pdf
This pdf file contains information about all the key-concepts learnt during the initial weeks of the Internship.


### You-Only-Speak_Once

[Original-Repo](https://github.com/sampathkethineedi/You-Only-Speak-Once/tree/master/fbank_net)

This is a slight modification of the above mentioned Original-Repo.
The recognise and the register functions have been modified. Resemblyzer has been used to obtain the embeddings instead of the model(FBankNet).
The threshold for similarity is kept as **0.82**. If the embedding's inner product is above the threshold, the speaker's voice is recognised.
 


