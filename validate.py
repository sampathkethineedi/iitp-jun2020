from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


MAX_LEN = 1024
MAX_SUMMARY_LEN = 200
BATCH_SIZE = 2 #Experimented with sizes of 2, 4, 8, and 16. 2 seems to be optimal
NUM_BEAMS = 4

params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0
        }
#load validation dataset in dataset_valid first

valid_loader = DataLoader(dataset_valid, **params)

def validate():
  model.eval()
  prediction = []
  reference = []
  with torch.no_grad():
      for _, data in enumerate(valid_loader, 0):
        source = tokenizer.batch_encode_plus(data['article'], truncation='only_first', max_length=MAX_LEN, pad_to_max_length=True,return_tensors='pt')
        target = tokenizer.batch_encode_plus(data['highlights'], truncation='only_first', max_length=MAX_LEN, pad_to_max_length=True,return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        y = target_ids.to(device, dtype = torch.long)
        ids = source_ids.to(device, dtype = torch.long)
        mask = source_mask.to(device, dtype = torch.long)

        generated_ids = model.generate(
            input_ids = ids,
            attention_mask = mask,
            max_length=MAX_SUMMARY_LEN, 
            num_beams=NUM_BEAMS,
            repetition_penalty=2.5, 
            no_repeat_ngram_size=4,
            early_stopping=True
            )#can experiment with different values of num_beams and penalties
        pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        ref = [tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)for r in y]
        if(_%100 == 0):
          print(f"{_} batches complete")

        prediction.extend(pred)
        reference.extend(ref)
  return prediction, reference

predictions, references = validate()
