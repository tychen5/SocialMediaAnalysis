def title_em_rule_inference(em_title, ori_title, taken_keywords, em_kw_li, classifier, classifier2, kw_map_di, show_di, title_kwem_di):
    """
    Perform rule inference on the title using the given classifiers and dictionaries.

    Args:
        em_title (str): Preprocessed title.
        ori_title (str): Original title.
        taken_keywords (list): List of taken keywords.
        em_kw_li (list): List of keywords in the title.
        classifier (function): First classifier function.
        classifier2 (function): Second classifier function.
        kw_map_di (dict): Keyword mapping dictionary.
        show_di (dict): Show dictionary.
        title_kwem_di (dict): Title keyword dictionary.

    Returns:
        list: Sorted list of inferred topics.
    """
    dfs = []
    res1 = classifier(ori_title, taken_keywords, multi_label=True)
    dfs.append(clean_df(res1, 1, thr=0.2))
    res2 = classifier2(ori_title, taken_keywords, multi_label=True)
    dfs.append(clean_df(res2, 2, thr=0.2))
    df_merge = reduce(lambda left, right: pd.merge(left, right, on=['labels'], how='inner'), dfs)
    df_merge['avg'] = df_merge.mean(axis=1, numeric_only=True)
    df_merge = df_merge[df_merge['avg'] > 0.49]
    df_merge = df_merge.sort_values(['avg'], ascending=False)
    df_merge['topic'] = df_merge['labels'].apply(get_topic, args=(kw_map_di, show_di))
    model_topic = df_merge['topic'].tolist()

    def ifexist(kw):
        if kw in em_title:
            return title_kwem_di[kw]

    em_topic = set(map(ifexist, em_kw_li))
    return sorted(set(model_topic) & em_topic, key=model_topic.index)


def post_keywords_inference_inhouse(title, selftext, id_key):
    """
    Perform keyword inference on a post using in-house models.

    Args:
        title (str): A post's original title.
        selftext (str): A post's original body.
        id_key (str): A post's id.

    Returns:
        list: Keywords predefined in dictionary.
    """
    take_df = saved_result_df[saved_result_df['id'] == id_key]
    if len(take_df) > 0:
        return take_df['keyword_li'].tolist()[0]
    else:
        lemmatizer = WordNetLemmatizer()
        taken_keywords, kw_cat_di, kw_map_di, show_di, topic2cat_di, title_kwem_di, latest_topics = pickle.load(open('/path/to/your/keyword_dict', 'rb'))
        em_kw_li = list(title_kwem_di.keys())
        classifier, classifier2, classifier3 = pickle.load(open('/path/to/your/models', 'rb'))
        end_punc = '!?.'
        full_doc = clean_text(title, selftext, 'original')
        paras = clean_text(title, selftext, 'paragraph')
        em_title = em_preprocess_algo(title, lemmatizer)
        dfs = []
        if len(full_doc) > 65:
            kw_li = long_post_model_inference(classifier, classifier2, classifier3, full_doc, paras, taken_keywords, dfs)
            topic_li_title = title_em_rule_inference(em_title, title, taken_keywords, em_kw_li, classifier, classifier2, kw_map_di, show_di, title_kwem_di)
            output = get_kw_name(kw_li, kw_cat_di, kw_map_di, show_di, latest_topics, topic_li_title, topic2cat_di)
            return output
        else:
            if len(full_doc) < 5:
                return []
            kw_li = short_post_model_inference(full_doc, taken_keywords, classifier, classifier2, end_punc)
            topic_li_title = title_em_rule_inference(em_title, title, taken_keywords, em_kw_li, classifier, classifier2, kw_map_di, show_di, title_kwem_di)
            output = get_kw_name(kw_li, kw_cat_di, kw_map_di, show_di, latest_topics, topic_li_title, topic2cat_di)
            return output