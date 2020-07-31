# Abstractive Summarizer
Automatic Abstractive Summarizer built by fine tuning HuggingFace's BartForConditionalGeneration model on the cnn-daily mail dataset.
APIs for the summarizer built using flask and a simple web app for convenient usage built with streamlit.

## How To use model (PyTorch model available):
```python
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model = AutoModelForSeq2SeqLM.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model.to(device)

MAX_LEN = 1024
SUMM_LEN = 250

article = <"text to be summarized (string)">

source = tokenizer.batch_encode_plus([article], max_length=max_len, pad_to_max_length=True,return_tensors='pt', truncation='only_first')
source_ids = source['input_ids']
source_mask = source['attention_mask']
ids = source_ids.to(device, dtype = torch.long)
mask = source_mask.to(device, dtype = torch.long)
cnn_generated_ids = cnn_model.generate(
          input_ids = ids,
          attention_mask = mask, 
          max_length=SUMM_LEN,
          num_beams=num_beams,
          repetition_penalty=2.5,
          early_stopping=True
          )

summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in cnn_generated_ids]

print(summary)
```

### Find description of available summarization models on transformers model hub [here](https://huggingface.co/yuvraj)

## To use the API:
Find API scripts and corresponding OpenAPI specifications in [this directory](https://github.com/ust-ilabs/iitp-jun2020/tree/nlp-yuvrajraghuvanshi/summAPI).

[SummAPI](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/summAPI/summAPI.py) is an API written in flask that takes articles sent with POST requests and returns a summary or an extreme summary (one-line summary) of the article.
SummAPI Swagger specification [here](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/summAPI/summAPI_spec.yaml) 

[SummAPIv2](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/summAPI/summAPIv2.py) is a newer version of [SummAPI](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/summAPI/summAPI.py) that improves on the previous version by adding a feature that finds similar articles to last summarized articles that were previously sumarized based on their summaries.
Find Swagger specification for this API [here](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/summAPI/summAPIv2_spec.yaml)

### Running the API(s)

#### Prerequisites

* Transformers
   * $ pip install transformers
* Flask
   * $ pip install flask
* Streamlit
   * $ pip install Streamlit     
* Pytorch
   * $ pip install torch

## Web App
Find scripts for the web apps in [frontend](https://github.com/ust-ilabs/iitp-jun2020/tree/nlp-yuvrajraghuvanshi/frontend) directory.

[Sumapp](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/frontend/summapp.py) is the old version of the web app, it has a simpl interface where you can paste the article in the provided text box, adjust the *number of beams* parameter as per your wish, and then click on *generate summary* or *generate extreme summary* buttons to request summaries from the API and display the summary.

[SummDocsimApp](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/frontend/SummDocsimApp.py) is the newer version of the web app, it is a superset of [Sumapp](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/frontend/summapp.py) with the additional feature of getting similar articles with the click of a button. Note that only articles that have been summarized previously using the 'Generate Summary' (and NOT 'Generate Extreme Summary') button will be searched for similarity checking and returned. Also, you have to summarize the article before requesting for similar articles, as the API essentially returns articles similar to the most refcently summarized articles under the hood.


### Running the Web Apps

1. Run the Flask API script
   1. $ python <path to summAPIv2.py> or $ python <path to summAPI.py>
1. Run web app
   1. $ streamlit run <path to SummDocsimApp.py> or $ streamlit run <path to summapp.py>
   1. The web app will open in your local machine's default browser after the above command is run
          
          
## Summarizing text/pdf documents- Document Summarizer

[Document Summarizer](https://github.com/ust-ilabs/iitp-jun2020/tree/nlp-yuvrajraghuvanshi/Document%20Summarizer) contains the python script for summarization of pdf/text files.
Download the [Document Summarizer script](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/Document%20Summarizer/General_Summarizer.py) and run it with
$ python <path to your downloaded script>

#### Prerequisites

* Transformers
   * $ pip install transformers   
* Pytorch
   * $ pip install torch
* Tika
   * $ pip install tika
   
The [script](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/Document%20Summarizer/General_Summarizer.py) takes as terminal input the path (relative or absolute both paths work) of the document to be summarized. Note that for files that are not pdf/text format, the behaviour is undefined and may produce runtime exceptions.
The outputs of the script:

#### Outputs of [Document Summarizer](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/Document%20Summarizer/General_Summarizer.py)

* Paragraph wise summaries (Note that a pagebreak character in a pdf file will be considered as a paragraph break)  
* Overall summary
