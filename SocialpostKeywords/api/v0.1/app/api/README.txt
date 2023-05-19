Goal: read inferenced results(pickles) and output user specified weeks' spotlights with document frequency and keyword scoring

Input: list of string({year}_{week#}), e.g., ['2022_8','2022_9']
Output: list of dicts(key=keywords) of dicts(key=doc_freq,kw_popularity ; value=int,float) e.g., [{'ap|uap': {'doc_freq': 28, 'kw_popularity': 90.656891215586},'switch|poe24 switch|poe switch us': {'doc_freq': 23,'kw_popularity': 86.14701160277585}, ... }, {'router|router output': {'doc_freq': 20, 'kw_popularity': 69.35482774272789}, ...}, ... ]