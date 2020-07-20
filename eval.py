import nlp

#get predictions and references before running

rouge_metric = nlp.load_metric("rouge")
score = rouge_metric.compute(predictions, references)
print(f"Rouge Score: {score}")
