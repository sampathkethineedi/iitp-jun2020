from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import random

class T5Model:

    def __init__(self, decoding_params):
        self.model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.decoding_params = decoding_params
        self.tokenizer = T5Tokenizer.from_pretrained(decoding_params["tokenizer"])
        self.sentence = ""

    def run_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)

        text = "paraphrase: " + self.sentence + " </s>"

        max_len = self.decoding_params["max_len"]

        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        if self.decoding_params["strategy"] == "Greedy Decoding":
            beam_outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=max_len
            )
        elif self.decoding_params["strategy"] == "Beam Search":
            beam_outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=max_len,
                num_beams=self.decoding_params["beams"],
                no_repeat_ngram_size=self.decoding_params["ngram"],
                early_stopping=True,
                temperature=self.decoding_params["temperature"],
                num_return_sequences=self.decoding_params["return_sen_num"]  # Number of sentences to return
            )
        elif self.decoding_params["strategy"] == "Top-p Top-k":
            beam_outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                do_sample=True,
                max_length=max_len,
                top_k=self.decoding_params["top_k"],
                top_p=self.decoding_params["top_p"],
                early_stopping=True,
                # temperature=decoding_params["temperature"],
                num_return_sequences=self.decoding_params["return_sen_num"]  # Number of sentences to return
            )

        return beam_outputs

    def checkDuplicate(self, paraphrase, decoding_params, temp):
        split_sentence = self.sentence.split(" ")

        paraphrase_set = set(paraphrase.split(" "))
        sentence_set = set(split_sentence)

        # print(paraphrase, len(paraphrase_set.intersection(sentence_set)))

        if len(paraphrase_set.intersection(sentence_set)) >= decoding_params["common"]:
            return False

        else:
            for line in temp:
                line_set = set(line.split(" "))
                # grammar_check = nlp(line)
                if len(paraphrase_set.intersection(line_set)) > len(split_sentence)//2:
                    return False
                # elif grammar_check._.has_grammar_error:
                #     return False

            return True

    def preprocess_output(self, model_output, tokenizer, temp, sentence, decoding_params, model):
        for line in model_output:
            paraphrase = tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if paraphrase.lower() != sentence.lower() and paraphrase not in temp:
                if decoding_params["strategy"] == "Top-k, Top-p sampling":
                    if self.checkDuplicate(paraphrase, decoding_params, temp):
                        temp.append(paraphrase)
                else:
                    temp.append(paraphrase)

        if decoding_params["strategy"] != "Greedy Decoding" and len(temp) < decoding_params["return_sen_num"]:
            temp1 = temp
            if decoding_params["strategy"] == "Top-k, Top-p sampling":
                sentence = self.sentence
            else:
                self.sentence = temp1[random.randint(0, len(temp1) - 1)]

            model_output = self.run_model()
            temp = self.preprocess_output(model_output, tokenizer, temp, sentence, decoding_params, model)
        return temp

    def forward(self, sentence):

        self.sentence = sentence
        model_output = self.run_model()

        paraphrases = []
        temp = []

        temp = self.preprocess_output(model_output, self.tokenizer, temp, sentence, self.decoding_params, self.model)

        # for i, line in enumerate(temp):
        #     paraphrases.append(f"{i + 1}. {line}")

        # return paraphrases
        return temp