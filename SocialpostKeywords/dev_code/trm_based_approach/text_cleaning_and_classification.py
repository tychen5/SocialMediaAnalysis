def clean_text(title, selftext, method='mpnet'):
    if method == 'mpnet':
        title = easy_clean_mpnet(title)
        selftext = easy_clean_mpnet(selftext)
        if len(title) > 1 and title[-1] in end_punc:
            return title + ' ' + selftext
        else:
            if len(title) > 1:
                return title + '. ' + selftext
            else:
                return selftext
    elif method == 'use':
        title_li = easy_clean_use(title)
        selftext_li = easy_clean_use(selftext)
        title_li.extend(selftext_li)
        return title_li.copy()
    else:
        return None


reddit_df['full_doc'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)


def easy_clean_use(doc):
    doc = html.unescape(doc)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", doc, re.I)
    try:
        email = email.group(1)
        doc = doc.replace(email, ' ')
    except AttributeError:
        pass

    doc = re.sub(r'https?://\S+|www\.\S+', ' ', doc)
    doc = emoji_pattern.sub(r'', doc)

    for punc in bad_puncli:
        doc = doc.replace(punc, ' ')
        doc = doc.replace('..', '.')

    doc_li = re.split(r'\n\n|\n\t', doc)
    clean_li = []

    for paragraph in doc_li:
        paragraph = re.sub(r'\n|\r|\t', ' ', paragraph)
        paragraph_li = paragraph.split(' ')
        paragraph_li = list(filter(None, paragraph_li))

        if len(paragraph_li) > 5:
            clean_li.append(" ".join(paragraph_li))

    return clean_li


reddit_df['paragraph_doc'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='use'), axis=1)
reddit_df.to_pickle('/path/to/your/processed_df')  # Replace with your path

all_full_doc = reddit_df['full_doc'].tolist()
all_paragraphs = reddit_df['paragraph_doc'].tolist()
all_time = reddit_df['created_utc'].tolist()
all_ori_title = reddit_df['title'].tolist()
all_ids = reddit_df['id'].tolist()
all_subgroup_li = reddit_df['subreddit'].tolist()

long_post_char = 65
model_lower_char = 23


def create_ds(all_ids, all_full_doc, all_ori_title):
    clf2_input_data = []
    clf2_input_id = []

    for prim_key, full_doc, title in zip(all_ids, all_full_doc, all_ori_title):
        full_doc = str(full_doc)
        title = str(title)

        if len(full_doc) < 5:
            continue

        full_doc = full_doc.replace('  ', ' ')

        if full_doc[0] == ' ':
            full_doc = full_doc[1:]

        if full_doc[-1] == ' ':
            full_doc = full_doc[:-1]

        if full_doc[-1] not in end_punc:
            full_doc = full_doc + '.'

        if len(full_doc) > model_lower_char:
            clf2_input_data.append(full_doc)
            clf2_input_id.append(prim_key)

            if len(title) < 5:
                clf2_input_data.append(title)
                clf2_input_id.append(prim_key + '_title')
        else:
            if len(title) < 5:
                clf2_input_data.append(title)
                clf2_input_id.append(prim_key + '_title')

    return clf2_input_data, clf2_input_id


clf1_input_data, clf1_input_id = create_ds(all_ids, all_full_doc, all_ori_title)
clf2_input_data, clf2_input_id = create_ds(all_ids, all_full_doc, all_ori_title)

clf3_input_data = {}
clf3_input_id = []

for prim_key, full_doc, paras in zip(all_ids, all_full_doc, all_paragraphs):
    if len(full_doc) > long_post_char:
        clf3_input_data[prim_key] = paras

def function2(clf2_input_id, all_full_doc, taken_keywords='enter_your_keywords', long_post_char=65, model_lower_char=23, check_point_save_path_func2='/path/to/your/checkpoint'):
    from transformers import pipeline
    from torch.utils.data import Dataset
    import pickle
    import gc, torch
    import nvidia_smi

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    # Load saved results or initialize empty dictionaries
    try:
        clf2_fulldoc_result_dict, clf2_shortpost_result_dict, clf2_title_result_dict, _ = pickle.load(open(check_point_save_path_func2, 'rb'))
    except FileNotFoundError:
        clf2_fulldoc_result_dict = {}
        clf2_shortpost_result_dict = {}
        clf2_title_result_dict = {}

    # Define custom dataset class
    class get_data(Dataset):
        def __init__(self, all_input_data):
            self.all_input_data = all_input_data

        def __len__(self):
            return len(self.all_input_data)

        def __getitem__(self, idx):
            return self.all_input_data[idx]

    dataset2 = get_data(all_full_doc)

    with torch.no_grad():
        classifier2 = pipeline("zero-shot-classification",
                               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", num_workers=1, device=0)
        i = 1
        run_idli = []

        try:
            for out2, prim_key, full_doc in zip(classifier2(dataset2, candidate_labels=taken_keywords, multi_label=True, batch_size=8), clf2_input_id, all_full_doc):
                run_idli.append(prim_key)

                # Classify long posts
                if (len(full_doc) > long_post_char) and ('_title' not in prim_key):
                    df = pd.DataFrame([out2['labels'], out2['scores']]).T
                    df.columns = ['labels', 'score']
                    df = df[df['score'] > 0.5]
                    clf2_fulldoc_result_dict[prim_key] = df

                # Classify short posts
                elif (len(full_doc) > model_lower_char) and ('_title' not in prim_key):
                    df = pd.DataFrame([out2['labels'], out2['scores']]).T
                    df.columns = ['labels', 'score']
                    df = df[df['score'] > 0.6]
                    clf2_shortpost_result_dict[prim_key] = df

                # Classify titles
                elif '_title' in prim_key:
                    df = pd.DataFrame([out2['labels'], out2['scores']]).T
                    df.columns = ['labels', 'score']
                    df = df[df['score'] > 0.2]
                    clf2_title_result_dict[prim_key] = df

                i = i + 1

                # Save results periodically
                if i % 10 == 0:
                    pickle.dump(obj=(clf2_fulldoc_result_dict, clf2_shortpost_result_dict, clf2_title_result_dict, run_idli),
                                file=open(check_point_save_path_func2, 'wb'))
                else:
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    if info.free * 1e-6 < 1024:
                        gc.collect()
                        torch.cuda.empty_cache()

        except ValueError as e:
            pickle.dump(obj=(clf2_fulldoc_result_dict, clf2_shortpost_result_dict, clf2_title_result_dict, run_idli),
                        file=open(check_point_save_path_func2, 'wb'))

    # Save final results
    pickle.dump(obj=(clf2_fulldoc_result_dict, clf2_shortpost_result_dict, clf2_title_result_dict, run_idli),
                file=open(check_point_save_path_func2, 'wb'))


if __name__ == '__main__':
    ray.init()
    ray.get([
        function2.remote(clf2_input_id, clf2_input_data, taken_keywords=taken_keywords),
        function1.remote(clf1_input_id, clf1_input_data, clf3_input_data, taken_keywords=taken_keywords)
    ])
    ray.shutdown()

class DatasetParagraph:
    def __init__(self, all_paragraphs_flat):
        """Initialize the DatasetParagraph class.

        Args:
            all_paragraphs_flat (list): A list of all paragraphs.
        """
        self.all_paragraphs_flat = all_paragraphs_flat

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: The number of paragraphs in the dataset.
        """
        return len(self.all_paragraphs_flat)

    def __getitem__(self, idx):
        """Get a paragraph from the dataset by index.

        Args:
            idx (int): The index of the paragraph to retrieve.

        Returns:
            str: The paragraph at the specified index.
        """
        return self.all_paragraphs_flat[idx]


dataset3 = DatasetParagraph(all_paragraphs_flat)


@ray.remote(num_gpus=1)
def func3(dataset3, taken_keywords):
    """Perform zero-shot classification on the dataset.

    Args:
        dataset3 (DatasetParagraph): The dataset containing paragraphs.
        taken_keywords (list): A list of candidate labels for classification.

    Returns:
        list: A list of classified sequences.
    """
    import torch
    from transformers import pipeline

    classifier3 = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        num_workers=1,
        device=0,
    )
    res = []
    for out3 in classifier3(
        dataset3, candidate_labels=taken_keywords, multi_label=True, batch_size=32
    ):
        res.append(out3["sequence"])

    return res

class DatasetFulldoc:
    def __init__(self, all_full_doc):
        self.all_full_doc = all_full_doc

    def __len__(self):
        return len(self.all_full_doc)

    def __getitem__(self, idx):
        return self.all_full_doc[idx]


dataset1 = DatasetFulldoc(all_full_doc)
dataset2 = DatasetFulldoc(all_full_doc)


@ray.remote(num_gpus=1)
def func1_2(dataset1, dataset2, taken_keywords):
    import torch
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli", num_workers=1, device=0)
    classifier2 = pipeline("zero-shot-classification",
                           model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", num_workers=1, device=0)

    res = []
    for out1, out2 in zip(classifier(dataset1, candidate_labels=taken_keywords, multi_label=True, batch_size=8),
                          classifier2(dataset2, candidate_labels=taken_keywords, multi_label=True, batch_size=8)):
        res.append(out1['sequence'])
    return res


if __name__ == '__main__':
    ray.init()
    ret1, ret2 = ray.get([func3.remote(dataset3=dataset3, taken_keywords=taken_keywords),
                          func1_2.remote(dataset1=dataset1, dataset2=dataset2, taken_keywords=taken_keywords)])
    ray.shutdown()

if __name__ == "__main__":
    processes = []
    p = mp.Process(target=func1_2, args=(dataset1, dataset2, taken_keywords))
    p.start()
    processes.append(p)
    p = mp.Process(target=func3, args=(dataset3, taken_keywords))
    processes.append(p)
    for p in processes:
        p.join()

mp.set_start_method('spawn')
p1 = Process(target=func1_2, args=(dataset1, dataset2, taken_keywords))
p1.start()
p2 = Process(target=func3, args=(dataset3, taken_keywords))
p2.start()
p1.join()
p2.join()

# The following code has been removed as it seems to be unnecessary:
# li = []
# li2 = ['123', [1, 2, 3], [1, 2, 3]]
# li.append(li2)
# li.append(li2)
# pd.DataFrame(li)

# The rest of the code has been left unchanged as it is not clear which parts are sensitive information.
# Please provide more context or specific instructions on which parts need to be modified.

import pandas as pd
from fuzzywuzzy import fuzz


def get_kw_name(kw_li, title_topics):
    """Get keyword names and categories from given lists.

    Args:
        kw_li (list): List of keywords.
        title_topics (list): List of title topics.

    Returns:
        list: Cleaned and categorized keyword names.
    """
    new_names = []
    taken_topics = []

    if not isinstance(kw_li, list) or not isinstance(title_topics, list):
        return []

    for kw in kw_li:
        # Replace sensitive information with dummy examples
        kw_map_di = {'dummy_key': 'dummy_value'}
        kw_cat_di = {'dummy_key': 'dummy_value'}
        show_di = {'dummy_key': 'dummy_value'}
        latest_topics = ['dummy_topic']
        topic2cat_di = {'dummy_key': 'dummy_value'}

        try:
            kw = kw_map_di[kw]
        except KeyError:
            kw = kw.split(',')[0].lower()
            kw = kw.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')

            if kw[0] == ' ':
                kw = kw[1:]
            if kw[-1] == ' ':
                kw = kw[:-1]

            try:
                cat_name = kw_cat_di[kw]
            except KeyError:
                cat_name = 'MISC.'
                uncategorized_kw_li.append(kw)

            try:
                show_name = show_di[kw]
            except KeyError:
                show_name = kw

            taken_topics.append(show_name)

            if show_name not in latest_topics:
                all_li = []
                for ltopic in latest_topics:
                    li = []
                    score = max(fuzz.token_set_ratio(show_name, ltopic),
                                fuzz.token_set_ratio(ltopic, show_name))
                    if score > 80:
                        li.append(ltopic)
                        li.append(score)
                        all_li.append(li)

                if len(all_li) > 0:
                    df = pd.DataFrame(all_li, columns=['latest_topic', 'score'])
                    max_score = df['score'].max()
                    df = df[df['score'] == max_score]

                    if len(df) == 1:
                        show_name = df['latest_topic'].tolist()[0]
                        taken_topics.append(show_name)

            new_names.append(cat_name + '\\' + show_name + '\\' + kw)

    for titl_topic in title_topics:
        if titl_topic not in taken_topics and titl_topic in latest_topics:
            cat_name = topic2cat_di[titl_topic]
            kw = 'complex'
            new_names.append(cat_name + '\\' + titl_topic + '\\' + kw)

    result_li = []
    cleaned_new_names = []

    for output in new_names:
        if output not in result_li:
            cleaned_new_names.append(output)
            result_li.append(output)

    return cleaned_new_names


# Replace sensitive information with dummy examples
reddit_df_ori = pd.DataFrame({'created_utc': ['dummy_data'],
                              'keyword_li_ori': ['dummy_data'],
                              'title_topic_li': ['dummy_data']})

reddit_df_ori['keyword_li_ori'] = reddit_df_ori['created_utc'].apply(match_kw)
reddit_df_ori['title_topic_li'] = reddit_df_ori['created_utc'].apply(match_title)
reddit_df_ori['keyword_li'] = reddit_df_ori.apply(
    lambda x: get_kw_name(x.keyword_li_ori, x.title_topic_li), axis=1)

uncategorized_kw_li = sorted(set(uncategorized_kw_li))

# Replace sensitive information with dummy examples
result_save_path = '/path/to/your/result.pkl'
reddit_df_ori.to_pickle(result_save_path)