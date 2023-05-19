
from torch.utils.data import Dataset
import re
import os
import pandas as pd
import pickle
from tqdm import tqdm

class GetData2(Dataset):
    def __init__(self, all_input_data):
        self.all_input_data = all_input_data

    def __len__(self):
        return len(self.all_input_data)

    def __getitem__(self, idx):
        return self.all_input_data[idx]

ocr_text_lili = ocr_df['ocr_text'].tolist()
ocr_needtest_idli = ocr_df['id'].tolist()

try:
    grammarpass_chk_point = pickle.load(open('path/to/your/grammar_chk_point', 'rb'))
    runned_id = list(grammarpass_chk_point.keys())
except FileNotFoundError:
    grammarpass_chk_point = {}
    runned_id = []

pattern = re.compile('[\W_]+')

for ocr_text_li, prim_key in tqdm(zip(ocr_text_lili, ocr_needtest_idli), total=len(ocr_text_lili)):
    if run_only_new and prim_key in runned_id:
        continue

    augment_textli = []
    for text in set(ocr_text_li):
        alphanum = pattern.sub('', text)
        if len(set(alphanum)) < 2:
            continue

        alpha = re.sub(r'[^a-zA-Z]', '', alphanum)
        if len(alpha) < 1:
            continue

        augment_textli.append(text)
        augment_textli.append(text[:-1])
        augment_textli.append(text[:-1].lower())
        augment_textli.append(text.lower())

    dataset1 = get_data1(augment_textli)
    dataset2 = GetData2(augment_textli)
    taken_li = []

    for i, (out_grammar1, out_grammar2, text) in enumerate(zip(grammar1(dataset1, batch_size=16, truncation=True, max_length=512), grammar2(dataset2, batch_size=16, truncation=True, max_length=512), augment_textli)):
        if i % 4 == 0:
            ori_text = text
            flag = 0

        if (flag >= 2) or (ori_text in taken_li):
            continue

        if (out_grammar1['label'] == 'LABEL_1') or (out_grammar2['label'] == 'LABEL_0'):
            flag = flag + 1

        if flag >= 2:
            taken_li.append(ori_text)

    grammarpass_chk_point[prim_key] = taken_li

grammar_chkpoint_df = pd.DataFrame.from_dict(grammarpass_chk_point.items())
grammar_chkpoint_df.columns = ['id', 'grammarpass_ocr']
ocr_df = pd.merge(left=ocr_df, right=grammar_chkpoint_df, on='id', how='left')
ocr_df['grammarpass_ocr'].value_counts(dropna=False)
ocr_df.sample(60)
grammar_pass_ocr = ocr_df[ocr_df['grammarpass_ocr'].apply(lambda x: len(x) > 0)]
grammar_pass_ocr.to_pickle('path/to/your/grammar_pass_df_path')

import gc
import torch

del grammar1
del grammar2
gc.collect()
torch.cuda.empty_cache()

use_gpu_id = 0
os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu_id)

grammar_pass_ocr = pd.read_pickle('path/to/your/grammar_pass_df_path')
grammar_pass_ocr = grammar_pass_ocr[grammar_pass_ocr['full_doc_len'] > 2]
grammar_pass_lili = grammar_pass_ocr['grammarpass_ocr'].tolist()
full_doc_li = grammar_pass_ocr['full_doc'].tolist()
id_li = grammar_pass_ocr['id'].tolist()

class GetFulldocData(Dataset):
    def __init__(self, all_input_data):
        self.all_input_data = all_input_data

    def __len__(self):
        return len(self.all_input_data)

    def __getitem__(self, idx):
        text = self.all_input_data[idx]
        if text[-1] == " ":
            return text[:-1]
        else:
            return text

from typing import List, Dict, Any

class GetLabelsData(Dataset):
    def __init__(self, all_input_data: List[str]):
        self.all_input_data = all_input_data
        self.augment_data = []

    def __len__(self) -> int:
        return len(self.all_input_data)

    def __getitem__(self, idx: int) -> List[str]:
        self.augment_data = []
        for text in self.all_input_data[idx]:
            self.augment_data.append(text[:512])
            self.augment_data.append(text[:-1][:512])
            self.augment_data.append(text[:-1].lower()[:512])
            self.augment_data.append(text.lower()[:512])
        return self.augment_data

data_fulldoc = get_fulldoc_data(full_doc_li)
data_labels = GetLabelsData(grammar_pass_lili)

@ray.remote(num_gpus=1)
def function1(data_fulldoc: List[str], data_labels: List[str], id_li: List[str] = id_li,
              model_chk_point1: str = 'model_checkpoint1_path', only_new: bool = True, use_gpu_id: int = 0) -> Dict[str, Any]:
    # Import statements and other code here

    return out_topicmodel1

@ray.remote(num_gpus=1)
def function2(data_fulldoc: List[str], data_labels: List[str], id_li: List[str] = id_li,
              model_chk_point2: str = 'model_checkpoint2_path', only_new: bool = True, use_gpu_id: int = 1) -> Dict[str, Any]:
    # Import statements and other code here

    return out_topicmodel2

if __name__ == '__main__':
    ray.init()
    out_topicmodel1, out_topicmodel2 = ray.get([
        function1.remote(data_fulldoc, data_labels),
        function2.remote(data_fulldoc, data_labels)
    ])
    ray.shutdown()

# Rest of the code here
