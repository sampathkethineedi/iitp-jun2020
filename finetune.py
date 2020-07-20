import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')#bart-large is also available, but bart-base is much smaller and switching to large may or may not be worth the extra computation
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.to(device)

MAX_LEN = 1024
MAX_SUMMARY_LEN = 200
BATCH_SIZE = 2 #Experimented with sizes of 2, 4, 8, and 16. 2 seems to be optimal
EPOCHS = 2 #going over 4 epochs may result in overfitting
LEARNING_RATE = 5e-5 #any value from 3e-4 to 5e-5 works reasonably well
ADAM_EPS = 1e-8
NUM_BEAMS = 4

params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0
        }

#load training dataset in dataset_train

train_loader = DataLoader(dataset_train, **params)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

def train(epoch):
  model.train()
  for _,data in enumerate(train_loader, 0):
    source = tokenizer.batch_encode_plus(data['article'], max_length=MAX_LEN, pad_to_max_length=True,return_tensors='pt', truncation='only_first')
    target = tokenizer.batch_encode_plus(data['highlights'], max_length=MAX_LEN, pad_to_max_length=True,return_tensors='pt', truncation='only_first')
    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()
    #
    y = target_ids.to(device, dtype = torch.long)
    y_ids = y[:, :-1].contiguous() #to create labels for target, we need to right shift the target ids
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 #padding tokens should be marked with -100 value so the model can output correct loss
    ids = source_ids.to(device, dtype = torch.long)
    mask = source_mask.to(device, dtype = torch.long)

    outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
    loss = outputs[0]
        
    if _%100==0:
      print(f'Epoch: {epoch}, Loss:  {loss}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)
