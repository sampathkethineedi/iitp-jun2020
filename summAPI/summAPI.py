import logging
logging.basicConfig(level=logging.ERROR)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from flask import Flask, jsonify
from flask import abort, request

tokenizer = AutoTokenizer.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model = AutoModelForSeq2SeqLM.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model.to(device)
xsumm_model = AutoModelForSeq2SeqLM.from_pretrained("yuvraj/xSumm")
xsumm_model.to(device)

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

MAX_LEN = 1024
SUMM_LEN = 250

def cnn_summarize(max_len, num_beams, art):
  cnn_model.eval()
  with torch.no_grad():
      source = tokenizer.batch_encode_plus([art], max_length=max_len, pad_to_max_length=True,return_tensors='pt', truncation='only_first')
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
      
      cnn_pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in cnn_generated_ids]
      return cnn_pred

app = Flask(__name__)

cnn_data = []
xsumm_data = []
@app.route("/summapi/v1.0.0")
def home():
    return "Welcome to summarizer API"

@app.route('/summapi/v1.0.0/summ', methods=['POST'])
def summ_post_article():
    if not request.json or not 'article' in request.json:
          abort(400)
    if(len(cnn_data) >= 10): #just to keep the list short. Can modify/remove this as per wish
      cnn_data.pop(0)
    art = request.json['article']
    max_len = MAX_LEN
    num_beams = 4
    if 'num_beams' in request.json:
      num_beams = request.json['num_beams']
      if(num_beams > 32):
        num_beams = 32
    cnn_pred = ""
    for chunk in chunkstring(art, 1000):
      cnn_pred = cnn_pred + cnn_summarize(max_len, num_beams, chunk)[0]
    cnn_pred = [cnn_pred]
    if not cnn_data:
      id = 0
    elif len(cnn_data) > 0:
      id = cnn_data[-1]['id'] + 1
    cnn_data.append({'id': id,
                 'article': art,
                 'summary': cnn_pred})
    return jsonify({'article': art, 'summary': cnn_pred}), 201



@app.route('/summapi/v1.0.0/xsumm', methods=['POST'])
def xsumm_post_article():
    if not request.json or not 'article' in request.json:
          abort(400)
    if(len(xsumm_data) >= 10): #just to keep the list short. Can modify/remove this as per wish
      xsumm_data.pop(0)
    xsumm_model.eval()
    art = request.json['article']
    max_len = 1024
    num_beams = 4
    if 'num_beams' in request.json:
      num_beams = request.json['num_beams']
      if(num_beams > 32):
        num_beams = 32
    with torch.no_grad():
      source = tokenizer.batch_encode_plus([art], max_length=max_len, pad_to_max_length=True,return_tensors='pt', truncation='only_first')
      source_ids = source['input_ids']
      source_mask = source['attention_mask']
      ids = source_ids.to(device, dtype = torch.long)
      mask = source_mask.to(device, dtype = torch.long)

      xsumm_generated_ids = xsumm_model.generate(
          input_ids = ids,
          attention_mask = mask, 
          max_length=SUMM_LEN,
          num_beams=num_beams,
          repetition_penalty=2.5,
          early_stopping=True
          )
      xsumm_pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in xsumm_generated_ids]
    if not xsumm_data:
      id = 0
    elif len(xsumm_data) > 0:
      id = xsumm_data[-1]['id'] + 1
    xsumm_data.append({'id': id,
                 'article': art,
                 'summary': xsumm_pred})
    return jsonify({'article': art, 'summary': xsumm_pred}), 201



# Optional GET methods. Not needed.
# @app.route('/summapi/v1.0.0/data', methods=['GET'])
# def get_data():
#   return jsonify({'data': data})

# @app.route('/summapi/v1.0.0/data/<int:view_id>', methods=['GET'])
# def view_select(view_id):
#     block = [block for block in data if block['id'] == view_id]
#     if len(task) == 0:
#         abort(404)
#     return jsonify({'block': block})

app.run()
