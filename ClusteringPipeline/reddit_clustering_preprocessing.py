def stanza_cut(trans_text, ori_text=None, stanza_li=None):
    """
    Tokenize the text using Stanza and return the best number of sentences and the best sentences.
    """
    stanza_len = []
    stanza_textli = []
    nlp = Pipeline(lang='en', processors="tokenize", use_gpu=False, verbose=False)
    st_doc = nlp(trans_text)
    st_doc_s = st_doc.sentences
    st_len = len(st_doc_s)
    sents = []
    for s in st_doc_s:
        sents.append(s.text)
    stanza_len.append(st_len)
    stanza_textli.append(sents)

    if ori_text:
        for i, lang_id in enumerate(stanza_li):
            if lang_id == 'en':
                nlp = None
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    nlp = Pipeline(lang=lang_id, processors="tokenize", use_gpu=False, verbose=False)
                except KeyError:
                    if i < len(stanza_li) - 1:
                        pass
                    else:
                        try:
                            nlp = MultilingualPipeline(max_cache_size=1, ld_batch_size=1, use_gpu=False)
                        except ValueError:
                            bad_list.append(ori_text)
                except stanza.pipeline.core.LanguageNotDownloadedError:
                    stanza.download(lang_id)
                    nlp = Pipeline(lang=lang_id, processors="tokenize", use_gpu=False, verbose=False)
                except Exception as e:
                    pass

                if nlp:
                    try:
                        st_doc = nlp(ori_text)
                    except ValueError:
                        bad_list.append(ori_text)
                    st_doc_s = st_doc.sentences
                    st_len = len(st_doc_s)
                    sents = []
                    for s in st_doc_s:
                        sents.append(s.text)
                    stanza_len.append(st_len)
                    stanza_textli.append(sents)

    best_num = max(stanza_len)
    idx = stanza_len.index(best_num)
    best_sent = stanza_textli[idx]
    return best_num, best_sent


def nltk_cut(trans_text, ori_text=None, name_li=None):
    """
    Tokenize the text using NLTK and return the best number of sentences and the best sentences.
    """
    nltk_len = []
    nltk_textli = []
    nl_sent = sent_tokenize(trans_text)
    nl_len = len(nl_sent)
    nltk_len.append(nl_len)
    nltk_textli.append(nl_sent)

    if ori_text:
        for name in name_li:
            nl_sent = None
            if name == 'english':
                try:
                    nl_sent = nltk.tokenize.sent_tokenize(ori_text, language=name)
                except LookupError:
                    pass

            if nl_sent:
                nl_len = len(nl_sent)
                nltk_len.append(nl_len)
                nltk_textli.append(nl_sent)

    best_num = max(nltk_len)
    idx = nltk_len.index(best_num)
    best_sent = nltk_textli[idx]
    return best_num, best_sent


def cut_into_sentences(ori_text, trans_text, language_id):
    """
    Cut the text into sentences using different tokenizers and return the best sentences.
    """
    only_en = False
    if ori_text == trans_text:
        only_en = True
        ori_text = clean_text(ori_text)
    else:
        ori_text = clean_text(ori_text)
        trans_text = clean_text(trans_text)
        if ori_text == trans_text:
            only_en = True
        else:
            spacy_multi2_doc = spacynlp_multi2(ori_text)

    if (ori_text == "") or (trans_text == ""):
        return []

    if only_en == False:
        twoword_li, fullname_li, stanza_li = define_language(ori_text, language_id, spacy_multi2_doc)

    best_numli = []
    best_sentli = []

    if only_en:
        spacy_num, spacy_sent = spacy_cut(ori_text)
    else:
        spacy_num, spacy_sent = spacy_cut(trans_text, ori_text, twoword_li)
    best_numli.append(spacy_num)
    best_sentli.append(spacy_sent)

    if only_en:
        stanza_num, stanza_sent = stanza_cut(ori_text)
    else:
        stanza_num, stanza_sent = stanza_cut(trans_text, ori_text, stanza_li)
    best_numli.append(stanza_num)
    best_sentli.append(stanza_sent)

    if only_en:
        nltk_num, nltk_sent = nltk_cut(ori_text)
    else:
        nltk_num, nltk_sent = nltk_cut(trans_text, ori_text, stanza_li)
    best_numli.append(nltk_num)
    best_sentli.append(nltk_sent)

    best_num = max(best_numli)
    idx = best_numli.index(best_num)
    best_sent = best_sentli[idx]

    take_sents = []
    for sent in best_sent:
        sent = str(sent)
        if (len(sent.split()) < 3 and only_en) or (len(set(sent)) < 4) or (len(sent.split()) == 2):
            pass
        else:
            take_sents.append(sent)

    return take_sents

import os
import pickle
import json
import requests
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained('xlm-roberta-base')
        self.multilabel_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.Mish(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, len(encode_reverse))
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        s1 = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        downs_topics = self.multilabel_layers(s1['pooler_output'])

        if output_hidden_states:
            return s1['hidden_states']
        elif output_attentions:
            return s1['attentions']
        elif output_hidden_states and output_attentions:
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics

# Replace the following paths with your own paths
model_path = "/path/to/your/model"
encoding_path = "/path/to/your/encoding_dict.pkl"
social_media_name = "enter_your_social_media_name"

device = 'cuda:0'
threshold = 0.5
encode_reverse = pickle.load(open(encoding_path, 'rb'))
encode_reverse = np.array(list(encode_reverse.values()))

category_model = MyBert()
loaded_state_dict = torch.load(model_path, map_location=device)
category_model.load_state_dict(loaded_state_dict)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['comment_li']
        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tmp = pt_batch['input_ids'].clone()
        pt_batch['input_ids'] = tmp.squeeze()
        tmp = pt_batch['attention_mask'].clone()
        pt_batch['attention_mask'] = tmp.squeeze()
        return pt_batch

    def __len__(self):
        return len(self.df)

# Replace df_comments with your own DataFrame
xlmr_dataset = Dataset(df_comments, tokenizer)
dataloader = DataLoader(
    xlmr_dataset, batch_size=64, num_workers=int(os.cpu_count()), shuffle=False
)

sig_func = nn.Sigmoid().to(device)
category_model.to(device).eval()

# Add your own code for processing the data and saving the results
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os
import gc
import numpy as np
import pandas as pd
import pickle

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['sent_translation']
        text = preprocess(text)
        text_len = self.tokenizer(text, truncation=True, max_length=512)
        text_len = sum(text_len['attention_mask'])
        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        pt_batch['input_ids'] = pt_batch['input_ids'].squeeze()
        pt_batch['attention_mask'] = pt_batch['attention_mask'].squeeze()
        return pt_batch, torch.tensor(text_len)

    def __len__(self):
        return len(self.df)

# Replace the following paths with your own paths
labeled_df_path = "/path/to/your/labeled_df.pkl"
model_path = "/path/to/your/model.pth"

labeled_df = pd.read_pickle(labeled_df_path)

tokenizer1 = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlmr_dataset = CustomDataset(labeled_df, tokenizer1)
dataloader1 = DataLoader(
    xlmr_dataset, batch_size=32, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
xlmt_dataset = CustomDataset(labeled_df, tokenizer2)
dataloader2 = DataLoader(
    xlmt_dataset, batch_size=16, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer3 = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
infoxlm_dataset = CustomDataset(labeled_df, tokenizer3)
dataloader3 = DataLoader(
    infoxlm_dataset, batch_size=16, num_workers=int(os.cpu_count()), shuffle=False
)

tokenizer4 = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
xlmalign_dataset = CustomDataset(labeled_df, tokenizer4)
dataloader4 = DataLoader(
    xlmalign_dataset, batch_size=16, num_workers=int(os.cpu_count()), shuffle=False
)

dataloader_li = [dataloader1, dataloader2, dataloader3, dataloader4]
device_li = ['cuda:0', 'cuda:1', 'cuda:1', 'cuda:1']

model1 = mybert()
loaded_state_dict = torch.load(model_path, map_location='cuda:0')
model1.load_state_dict(loaded_state_dict)

config2 = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
model2 = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=config2)

config3 = AutoConfig.from_pretrained("microsoft/infoxlm-base")
model3 = AutoModel.from_pretrained("microsoft/infoxlm-base", config=config3)

config4 = AutoConfig.from_pretrained("microsoft/xlm-align-base")
model4 = AutoModel.from_pretrained("microsoft/xlm-align-base", config=config4)

model_li = [model1, model2, model3, model4]
weight_li = [4, 0.5, 1, 1.5]

# Replace the following path with your own path
output_path = "/path/to/your/output.pkl"

def extract_features_emb(dataloader_li, model_li, device_li, weight_li):
    # Your implementation here
    pass

def if_bad(nparray):
    if pd.isna(nparray[0]):
        return 0
    else:
        return 1

labeled_df['take'] = labeled_df['Embeddings'].apply(if_bad)
labeled_df = labeled_df[labeled_df['take'] == 1]
labeled_df = labeled_df.drop(['take'], axis=1)
pickle.dump(obj=labeled_df, file=open(output_path, 'wb'))