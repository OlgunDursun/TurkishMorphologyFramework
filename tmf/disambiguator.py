# %%


import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, BertTokenizer
from transformers import PreTrainedModel, BertModel, BertConfig
import json
from .analyzer import Analyzer


# Use device = 'mps' if running on Macos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = 'tmf/resources/local_full_dataset.json'

with open(dataset, "r", encoding="utf8") as f:
    raw_data = json.load(f)

model_name = "dbmdz/bert-base-turkish-cased"

tokenizer = BertTokenizer.from_pretrained(model_name)

class MultiTaskDataset(Dataset):
    def __init__(self, sentences, pos_labels, morph_labels, tokenizer, pos_label_map, morph_label_map, max_len=512):
        self.sentences = sentences
        self.pos_labels = pos_labels
        self.morph_labels = morph_labels
        self.tokenizer = tokenizer
        self.pos_label_map = pos_label_map
        self.morph_label_map = morph_label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        pos_labels = self.pos_labels[item]
        morph_labels = self.morph_labels[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length'
        )

        pos_label_ids = [-100]*self.max_len
        morph_label_ids = [-100]*self.max_len
        cur_pos_labels = [self.pos_label_map[label] for label in pos_labels]
        cur_morph_labels = [self.morph_label_map[label] for label in morph_labels]
        pos_label_ids[:len(cur_pos_labels)] = cur_pos_labels
        morph_label_ids[:len(cur_morph_labels)] = cur_morph_labels

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'pos_labels': torch.tensor(pos_label_ids, dtype=torch.long),
            'morph_labels': torch.tensor(morph_label_ids, dtype=torch.long),
        }


class MultiTaskModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.pos_classifier = torch.nn.Linear(config.hidden_size, config.num_pos_labels)
        self.morph_classifier = torch.nn.Linear(config.hidden_size, config.num_morph_labels)

    def forward(self, input_ids, attention_mask, pos_labels=None, morph_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pos_logits = self.pos_classifier(outputs[0])
        morph_logits = self.morph_classifier(outputs[0])

        if pos_labels is not None and morph_labels is not None:
            loss_fct = CrossEntropyLoss()
            pos_loss = loss_fct(pos_logits.view(-1, self.config.num_pos_labels), pos_labels.view(-1))
            morph_loss = loss_fct(morph_logits.view(-1, self.config.num_morph_labels), morph_labels.view(-1))
            loss = pos_loss + morph_loss
            return loss, pos_logits, morph_logits
        else:
            return pos_logits, morph_logits


pos_label_list = sorted(list(set(tag for example in raw_data for tag in example['pos_labels'])))
pos_label_map = {label: i for i, label in enumerate(pos_label_list)}

morph_label_list = sorted(list(set(tag for example in raw_data for tag in example['morph_labels'])))
morph_label_map = {label: i for i, label in enumerate(morph_label_list)}

num_pos_labels = len(pos_label_list)
num_morph_labels = len(morph_label_list)



config = BertConfig.from_pretrained("dbmdz/bert-base-turkish-cased")
config.num_pos_labels = num_pos_labels
config.num_morph_labels = num_morph_labels

model = MultiTaskModel.from_pretrained("tmf/resources/multihead_pos_morph_6.model", config=config).to(device)

def infer(inputter):
    sentence = inputter["sentence"]
    pos_subset = inputter["pos_subset"]
    morph_subset = inputter["morph_subset"]

    predicted_pos_labels = []
    predicted_morph_labels = []

    for i in range(len(sentence)):
        # Tokenize the word and map the labels to IDs
        encoding = tokenizer.encode_plus(
            sentence[i],
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            padding='max_length'
        )

        # Create a PyTorch tensor for the input data
        input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(device)
        attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(device)

        # Perform inference
        # Perform inference
        with torch.no_grad():
            pos_logits, morph_logits = model(input_ids, attention_mask=attention_mask)

            # Create masks for invalid logits
            pos_mask = torch.ones_like(pos_logits)
            morph_mask = torch.ones_like(morph_logits)
            if pos_subset[i] != []:
                for label in pos_label_map:
                    if label not in pos_subset[i]:
                        pos_mask[0][0][pos_label_map[label]] = float('-inf')
                for label in morph_label_map:
                    if label not in morph_subset[i]:
                        morph_mask[0][0][morph_label_map[label]] = float('-inf')
            # Apply the masks
            pos_logits += pos_mask
            morph_logits += morph_mask

            predicted_pos_label = torch.argmax(F.softmax(pos_logits, dim=2), dim=2)[0][0]
            predicted_morph_label = torch.argmax(F.softmax(morph_logits, dim=2), dim=2)[0][0]

        # Convert the predicted label IDs back to labels
        predicted_pos_label = pos_label_list[predicted_pos_label.item()]
        predicted_morph_label = morph_label_list[predicted_morph_label.item()]

        # Check if the predicted labels are in the respective subset lists
        if predicted_pos_label in pos_subset[i] and predicted_morph_label in morph_subset[i]:
            predicted_pos_labels.append(predicted_pos_label)
            predicted_morph_labels.append(predicted_morph_label)
        else:
            # Find the first matching morph label for the predicted pos label
            matching_indexes = [index for index, pos_label in enumerate(pos_subset[i]) if pos_label == predicted_pos_label]
            if matching_indexes:
                matching_morph_label = morph_subset[i][matching_indexes[0]]
                predicted_pos_labels.append(predicted_pos_label)
                predicted_morph_labels.append(matching_morph_label)
            else:
                # Append null or any default values if there is no match
                predicted_pos_labels.append(predicted_pos_label)
                predicted_morph_labels.append(None)

    return sentence, predicted_pos_labels, predicted_morph_labels



def convert_tag(pos_tag):
    ud_pos_mapping = {
    'Adj': 'ADJ',
    'adj': 'ADJ',
    'adjective': 'ADJ',
    'Postp': 'ADP',
    'postp': 'ADP',
    'Adv': 'ADV',
    'adv': 'ADV',
    'adverb': 'ADV',
    'Ques': 'AUX',
    'ques': 'AUX',
    'Conj': 'CCONJ',
    'conj': 'CCONJ',
    'Det': 'DET',
    'det': 'DET',
    'Interj': 'INTJ',
    'interj': 'INTJ',
    'Noun': 'NOUN',
    'noun': 'NOUN',
    'Num': 'NUM',
    'num': 'NUM',
    'number': 'NUM',
    'Pron': 'PRON',
    'pron': 'PRON',
    'Prop': 'PROPN',
    'prop': 'PROPN',
    'proper_noun': 'PROPN',
    'Punc': 'PUNCT',
    'punc': 'PUNCT',
    'punctuation': 'PUNCT',
    'Conj-dummy': 'SCONJ',
    'Verb': 'VERB',
    'verb': 'VERB',
    'Unknown': 'X',
    'Dup': 'ADV',
    'Pause': 'INTJ',
    }
    return ud_pos_mapping[pos_tag]