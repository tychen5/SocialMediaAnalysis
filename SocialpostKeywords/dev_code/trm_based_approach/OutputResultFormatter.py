def get_output_result(title_topic, long_post_topic, long_post_keyphrase, short_post_topic, short_post_keyphrase, topic2cat_di, latest_topics, kw_map_di):
    """
    Function to process and format the output result based on the input topics and keyphrases.
    """
    # Determine the type of post (long or short) and copy the corresponding topics and keyphrases
    if isinstance(long_post_topic, list):  # it is a long post
        take_topic = long_post_topic.copy()
        take_keyphrase = long_post_keyphrase.copy()
    elif isinstance(short_post_topic, list):  # it is a short post
        take_topic = short_post_topic.copy()
        take_keyphrase = short_post_keyphrase.copy()
    else:  # only title
        take_topic = []
        take_keyphrase = []

    # Add title topics to the list if not already present
    if isinstance(title_topic, list):
        for t in title_topic:
            if t not in take_topic:
                take_topic.append(t)
                take_keyphrase.append('complex')

    new_names = []
    take_topic = [x.lower() for x in take_topic]

    return cleaned_new_names



def mongo_to_df(filter_day=7):
    """
    Function to fetch data from MongoDB and convert it to a DataFrame.
    """
    client = pymongo.MongoClient(host='your_host', username='your_username', password='your_password', port=your_port,
                                 authSource='your_authSource', authMechanism='your_authMechanism')
    db = client.reddit_ui
    lastweekday = (dt.datetime.today() - dt.timedelta(days=filter_day)).replace(hour=0, minute=0, second=0, microsecond=0)
    reddit_column = ['id', 'subreddit', 'author', 'created_utc', 'year', 'week', 'date', 'full_link', 'link_flair_text', 'title', 'selftext', 'productline', 'productname', 'zsl_tags_level1', 'zsl_tags_level2', 'zsl_tags_level3', 'sentiment_type']
    projection = {i: 1 for i in reddit_column}
    projection['_id'] = 0
    prod_collection = db.posts_today.find({'created_utc': {'$gte': lastweekday}}, projection).sort('created_utc', pymongo.DESCENDING)
    df_reddit = pd.DataFrame(prod_collection)
    prod_collection = db.posts_today_community.find({'created_utc': {'$gte': lastweekday}}, projection).sort('created_utc', pymongo.DESCENDING)
    df_community = pd.DataFrame(prod_collection)
    df = pd.concat([df_reddit, df_community])
    df = df.sort_values(['created_utc'], ascending=False)
    return df
