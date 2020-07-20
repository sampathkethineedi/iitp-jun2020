## Description
Automatic Abstractive Summarization model built by fine tuning HuggingFace's BartForConditionalGeneration on the cnn-daily mail dataset.
An API for the same built using flask and a simple front end built using streamlit.

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
Find API script and corresponding OpenAPI specification in [this directory](https://github.com/ust-ilabs/iitp-jun2020/tree/nlp-yuvrajraghuvanshi/summAPI)

## Front end
Run flask API on localhost and subsequently run [summapp.py](https://github.com/ust-ilabs/iitp-jun2020/blob/nlp-yuvrajraghuvanshi/frontend/summapp.py) script using streamlit. Command for the same: *streamlit run sumapp.py*.
