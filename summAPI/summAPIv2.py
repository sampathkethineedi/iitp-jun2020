import logging
logging.basicConfig(level=logging.ERROR)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead
import torch
import torch.nn.functional as F
import numpy as np
import faiss
from flask import Flask, jsonify
from flask import abort, request

tokenizer = AutoTokenizer.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model = AutoModelForSeq2SeqLM.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model.to(device)
xsumm_model = AutoModelForSeq2SeqLM.from_pretrained("yuvraj/xSumm")
xsumm_model.to(device)

docsim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
docsim_model = AutoModelWithLMHead.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
docsim_model.to(device)

MAX_LEN = 1024
SUMM_LEN = 250
CHUNK_SIZE = 2000
EMBEDDING_SIZE = 30522

#Function for seperating articles into chunks
#each chunk is less than a fixed 'length' and ends with a period except for perhaps the last chunk
def chunkstring(string, length):
  string = " ".join(string.splitlines())
  chunks = []
  end_index = length-1
  start_index = 0
  while (True):
    if (end_index >= len(string)):
      chunks.append(string[start_index:])
      break
    while (string[end_index] != '.' and end_index > start_index):
      end_index = end_index - 1
    if (end_index == start_index):
      chunks.append(string[start_index : start_index+length+1])
      start_index = start_index+length
    elif (string[end_index] == '.'):
      chunks.append(string[start_index:end_index+1])
      start_index = end_index+1
    end_index = start_index + length - 1
  return chunks

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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def normalize(embeddings):
  for idx, embedding in enumerate(embeddings):
    embeddings[idx] = F.normalize(embedding, dim = 0)
  return embeddings

def encode(sentences_list):
  encoded_input = docsim_tokenizer(sentences_list, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
  docsim_model.eval()
  with torch.no_grad():
      model_output = docsim_model(**encoded_input)

  #Perform pooling. In this case, mean pooling
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  return sentence_embeddings

app = Flask(__name__)

cnn_data = []
xsumm_data = []

documents = []

d = encode(["Filler Sentence"])[0].size()[0]
index = faiss.IndexFlatIP(EMBEDDING_SIZE)


@app.route("/summapi/v2.0.0")
def home():
    return "Welcome to summarizer API"

@app.route('/summapi/v2.0.0/summ', methods=['POST'])
def summ_post_article():
    if not request.json or not 'article' in request.json:
          abort(400)
    if(len(cnn_data) >= 50): #just to keep the list short. Can modify/remove this as per wish
      cnn_data.pop(0)
    art = request.json['article']
    max_len = MAX_LEN
    num_beams = 4
    if 'num_beams' in request.json:
      num_beams = request.json['num_beams']
      if(num_beams > 32):
        num_beams = 32
    cnn_pred = ""
    for chunk in chunkstring(art, CHUNK_SIZE):
      cnn_pred = cnn_pred + cnn_summarize(max_len, num_beams, chunk)[0]
    cnn_pred = [cnn_pred]
    if not cnn_data:
      id = 0
    elif len(cnn_data) > 0:
      id = cnn_data[-1]['id'] + 1
    cnn_data.append({'id': id,
                 'article': art,
                 'summary': cnn_pred})
    new_embedding = normalize(encode(cnn_pred))
    documents.append(new_embedding)
    index.add(new_embedding.cpu().numpy())
    return jsonify({'article': art, 'summary': cnn_pred}), 201


@app.route('/summapi/v2.0.0/xsumm', methods=['POST'])
def xsumm_post_article():
    if not request.json or not 'article' in request.json:
          abort(400)
    if(len(xsumm_data) >= 50): #just to keep the list short. Can modify/remove this as per wish
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


def search(encoded_query, k=1):
  if(k > index.ntotal):
    k = index.ntotal
  D, I = index.search(encoded_query.cpu().numpy(), k)
  scores = D[0]
  query_results = [cnn_data[_id]['article'] for _id in I[0]]
  return list(zip(query_results, scores))


@app.route('/summapi/v2.0.0/docsim', methods=['GET'])
def get_data():
  query_embedding = documents[-1]
  cos_threshold = float(request.args.get('cos_tsld'))
  if cos_threshold is None:
    abort(400)
  knns = search(query_embedding, index.ntotal)
  knns = [element for element in knns if element[1] >= cos_threshold and element[1] < 0.99]
  knns = list(set(knns))
  knn_articles = [element[0] for element in knns]
  knn_scores = [int(element[1]*100) for element in knns]
  return jsonify({'data': knn_articles, 'scores': knn_scores}), 200

app.run()
