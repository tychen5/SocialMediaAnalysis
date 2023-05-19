
import itertools
import math
import pickle
import numpy as np
from fuzzywuzzy import fuzz
from kneed import KneeLocator
from tqdm import tqdm

def df_tf_score(kw_li, doc_li):
    df_score_li = []
    tf_score_li = []
    for kw in kw_li:
        df_count = 0
        tf_score_kw = 0
        for doc in doc_li:
            doc = doc.lower()
            tsr = fuzz.token_set_ratio(kw, doc)
            pr = fuzz.partial_ratio(doc, kw)
            df_score = np.mean([tsr, pr])
            if df_score > 80:
                df_count += 1
            tf_score = np.mean([fuzz.token_sort_ratio(kw, doc), tsr, fuzz.ratio(doc, kw), pr])
            if tf_score > 20:
                tf_score_kw = tf_score_kw + tf_score * 0.01
        df_score_li.append(df_count)
        tf_score_li.append(np.log1p(tf_score_kw))
    return np.mean(df_score_li), np.mean(tf_score_li)

for i in tqdm(mpnet_sample_postsdf.index):
    taken_idx = []
    use_kw_df = build_kw_df(use_sample_postsdf, mpnet_sample_postsdf, i)
    use_year, use_week = use_sample_postsdf.loc[i, 'year'], use_sample_postsdf.loc[i, 'week']
    mpnet_year, mpnet_week = mpnet_sample_postsdf.loc[i, 'year'], mpnet_sample_postsdf.loc[i, 'week']
    use_kw_df[['keyword_li', 'similar_scoreli', 'corr_idxli']] = use_kw_df.apply(
        lambda x: calc_similar(x.keyword, x.similarity, use_kw_df), axis=1, result_type='expand')
    tmp = use_kw_df[~use_kw_df['keyword_li'].isna()]
    combined_idx = tmp['corr_idxli'].tolist()
    combined_idx_li = list(itertools.chain.from_iterable(combined_idx))
    y = use_kw_df['similarity'].tolist()
    y = sorted(y, reverse=False)
    x = [i for i in range(len(y))]
    kneedle = KneeLocator(x, y, curve="convex", direction="increasing", online=True, S=3)
    sim_thrli = kneedle.all_knees_y
    for sim_thr in sim_thrli:
        use_kw_df['take_kw'] = use_kw_df.apply(
            lambda x: valid_kw(x.similar_scoreli, x.similarity, x.name, sim_thr, combined_idx_li), axis=1)
        take_kw = use_kw_df[use_kw_df['take_kw'] == 1]
        if len(take_kw) > 20:
            take_kw[['kw_name', 'kw_group']] = take_kw.apply(
                lambda x: find_common_str(x.keyword, x.keyword_li), axis=1, result_type='expand')
            week_docs = mpnet_sample_postsdf['doc_mpnet'].tolist()[0]
            take_kw[['df', 'tf']] = take_kw.apply(lambda x: df_tf_score(x.kw_group, week_docs), axis=1,
                                                  result_type='expand')
            take_kw['kw_score'] = take_kw['df'] * take_kw['tf']
            take_kw = take_kw.sort_values(['kw_score'], ascending=False)
            y = take_kw['kw_score'].tolist()
            y = sorted(y, reverse=False)
            x = [i for i in range(len(y))]
            kneedle = KneeLocator(x, y, curve="convex", direction="increasing", online=True, S=3)
            score_thrli = kneedle.all_knees_y
            for score_thr in score_thrli:
                overall_kw_df = take_kw[take_kw['kw_score'] > score_thr]
                if len(overall_kw_df) > 10:
                    overall_kw_df = overall_kw_df[['kw_name', 'kw_group', 'df', 'tf', 'kw_score']]
                    output_di = {}
                    for idx, row in overall_kw_df.iterrows():
                        all_kwli = row['kw_group'].copy()
                        all_kwli.extend(row['kw_name'].split('/'))
                        final_kw = sorted(set(all_kwli), key=len)
                        kw_name = "|".join(final_kw)
                        output_di[kw_name] = {}
                        output_di[kw_name]['doc_freq'] = math.ceil(row['df'])
                        output_di[kw_name]['kw_popularity'] = row['kw_score']
                    pickle.dump(obj=(overall_kw_df, output_di),
                                file=open('../output/' + str(use_year) + '_' + str(mpnet_week) + '.pkl', 'wb'))

def compare_list_str_sim(leftli, rightli):
    lefts_manyrights_fuzzscoreli = []
    lefts_manyrights_diffscoreli = []
    for leftstr in leftli:
        for rightstr in rightli:
            fuzz_score = np.mean([fuzz.token_sort_ratio(leftstr, rightstr),
                                  fuzz.token_set_ratio(leftstr, rightstr),
                                  fuzz.ratio(rightstr, leftstr),
                                  fuzz.partial_ratio(rightstr, leftstr)])
            s = difflib.SequenceMatcher(lambda x: x == " ", current_keyword, candidate_kw)
            diff_score = s.ratio()
            lefts_manyrights_fuzzscoreli.append(fuzz_score)
            lefts_manyrights_diffscoreli.append(diff_score)
    if (np.mean(lefts_manyrights_fuzzscoreli) > 80) and (np.mean(lefts_manyrights_diffscoreli) > 0.6):
        return 1
    else:
        return 0

taken_idx = []

import itertools
import numpy as np
import nltk
from kneed import KneeLocator
from fuzzywuzzy import fuzz


def calc_merge(current_kw_li, current_idx, all_df):
    """
    Calculate the merge of keywords and indices.

    Args:
        current_kw_li (list): List of current keywords.
        current_idx (int): Current index.
        all_df (DataFrame): DataFrame containing all data.

    Returns:
        list or float: Merged keywords list or NaN if the length of merge_index is less than or equal to 1.
    """
    compare_df = all_df.drop(taken_idx)
    compare_df = compare_df[compare_df.index > current_idx]
    compare_df['similar'] = compare_df['kw_group'].apply(
        compare_list_str_sim, args=(current_kw_li,))
    take_df = compare_df[compare_df['similar'] == 1]
    merge_index = take_df.index.tolist()
    taken_idx.extend(merge_index)
    merge_index.append(current_idx)

    if len(merge_index) > 1:
        merge_kw_li = take_df['kw_group'].tolist()
        merge_kw_li = list(itertools.chain.from_iterable(merge_kw_li))
        merge_kw_li.extend(current_kw_li)
        return sorted(set(merge_kw_li), key=len)
    else:
        return np.nan


df['merged_kws'] = df.apply(
    lambda x: calc_merge(x.kw_group, x.name), axis=1)
