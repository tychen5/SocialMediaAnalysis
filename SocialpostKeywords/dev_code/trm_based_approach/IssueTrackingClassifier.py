
def classify_function(element):
    """Classify the given element and update the application_clf_di dictionary."""
    try:
        prefix = main_rn_dict[element]
    except KeyError:
        return 0

    application = prefix.split(" ")[0]
    li = application_clf_di[application]

    try:
        element = kw_cat_di[element]
        li.append(element)
    except KeyError:
        li.append(element)

    application_clf_di[application] = li
    return 1


tmp = list(map(classify_function, all_issue_rn))

knee_dict = {}  # key=issue name, value=list: need what kind of kw occur simultaneously

for application, li in application_clf_di.items():
    di = dict(Counter(li))
    all_val = sorted(list(di.values()))
    kneedle = KneeLocator([i for i in range(len(all_val))], all_val, S=1, curve="convex", direction="increasing",
                           online=True)

    try:
        kneedle.plot_knee()
    except TypeError:
        pass

    knee_li = kneedle.all_knees_y

    if len(knee_li) == 0:  # no knee point
        for kw in set(li):
            knee_dict[kw] = []
    elif len(knee_li) == 1:  # only 1 knee point
        knee_point = knee_li[0]
        for k, v in di.items():
            if v > knee_point:
                vers_li = version_dict[application].copy()
                vers_li.extend(['UniFi ' + application + ' Application', 'Ubiquiti ' + application + ' Application',
                                application + ' Application'])
                knee_dict[k] = vers_li
            else:
                knee_dict[k] = []
    else:  # at least two knee points
        knee_li = sorted(knee_li)
        big_knee_point = knee_li[-1]
        small_knee_point = knee_li[-2]
        for k, v in di.items():
            if v > big_knee_point:
                vers_li = version_dict[application].copy()
                vers_li.extend(['UniFi ' + application + ' Application', 'Ubiquiti ' + application + ' Application'])
                knee_dict[k] = vers_li
            elif v > small_knee_point:
                vers_li = version_dict[application].copy()
                vers_li.extend(['UniFi ' + application + ' Application', 'Ubiquiti ' + application + ' Application',
                                application + ' Application'])
                knee_dict[k] = vers_li
            else:
                knee_dict[k] = []

pickle.dump(obj=(taken_keywords, kw_cat_di, all_kw_df, application_clf_di, version_dict, knee_dict),
            file=open('path/to/your/pickle_file', 'wb'))

uncategorized_kw_li = []


def get_kw_name(kw_li):
    """Get the keyword names from the given list."""
    new_names = []
    for kw in kw_li:
        try:
            release_note = kw_cat_di[kw]
        except KeyError:
            release_note = kw

        try:
            prefix = main_rn_dict[release_note]
            need_related_words = knee_dict[release_note]
            for word in need_related_words:
                if word in kw_li:
                    new_names.append(prefix + '\\' + kw)
        except KeyError:
            uncategorized_kw_li.append(kw)

    return sorted(set(new_names))


reddit_df_ori['keyword_li'] = reddit_df_ori['keyword_li_ori'].apply(get_kw_name)
reddit_df_ori.to_pickle('path/to/your/result_save_path')
