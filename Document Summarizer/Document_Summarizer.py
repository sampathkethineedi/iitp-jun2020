from tika import parser

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model = AutoModelForSeq2SeqLM.from_pretrained("yuvraj/summarizer-cnndm")
cnn_model.to(device)

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

def load_text(doc_path):
  text = ""
  doc_obj = open(doc_path, "r")
  ext = os.path.splitext(doc_path)[-1].lower()
  if (ext == ".pdf"):
    parsed = parser.from_file(doc_path)
    text = parsed["content"]
  elif (ext == ".txt"):
    text = doc_obj.read()
  return text

def paragraphSummary(text):
  pSumm = []
  article = list(filter(lambda x : x != '', text.split('\n\n')))
  for art in article:
    summary = summarize(art)
    pSumm.append(summary[0])
  return pSumm

def fullSummary(text):
  final_summary = ""
  for chunk in chunkstring(text, 4000):
    final_summary = final_summary + summarize(chunk)[0]
  return final_summary

doc_path = input("Enter document path")

text = load_text(doc_path)
text = text.strip()
print("Paragraph wise summary:")
pSumm = paragraphSummary(text)
print(pSumm)

print("\n\nOverall summmary:")
full_summary = fullSummary(text)
print(full_summary)
