import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk, datetime, re, warnings, os, pickle, sklearn
from fuzzywuzzy import fuzz, process
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from scipy import interp

def ModelIt(fromUser  = 'Default', brand='', product='', model='', raw_data = []):
  #Clean up the input
    if len(raw_data) == 0:
        print('ERROR')
        return

    embeddings = np.concatenate([np.load('recall/embedding_chunks/'+chunk) for chunk in os.listdir('recall/embedding_chunks')], axis = 0)
    embedded_data = pd.concat([raw_data.iloc[range(len(embeddings)), :], pd.DataFrame(embeddings)], axis=1)
    embedded_data.columns = embedded_data.columns.astype(str)

    def find_search_match(data, string, term_type, threshold=80):
        match = [s.lower() for s in list(set(embedded_data[term_type].values)) if isinstance(s, str)]
        candidate_products = pd.DataFrame(process.extract(string.lower(), match, 
                                                           scorer=fuzz.token_set_ratio),
                                          columns=['field', 'score'])
        rank = candidate_products.sort_values(by='score', ascending=False)
        if len(rank[rank['score'] == 100]) >= 5:
            selection = rank[rank['score'] == 100]
        else:
            results = min(len(rank[rank['score'] >= threshold]), 5)
            selection = rank[rank['score'] >= threshold].loc[0:results, :]
        if selection.empty:
            return("No match.")
        return(embedded_data[embedded_data[term_type].apply(lambda x: x.lower() in 
                                                            [s.lower() for s in selection['field'] if isinstance(s, str)]
                                                           if isinstance(x, str) else False)])

    product_pass = find_search_match(embedded_data, product, 'clean_product') if product else embedded_data
    brand_pass = find_search_match(product_pass, brand, 'clean_brand') if brand else product_pass
    model_pass = find_search_match(brand_pass, model, 'model_name_or_number', 100) if model else brand_pass

    logreg_model = pickle.load(open('recall/trained_model.sav', 'rb'))

    recall_proba = 0

    if (model_pass['0'].values > 0).sum() < len(model_pass):
        recall_proba = np.mean(logreg_model.predict_proba(model_pass[[str(i) for i in list(range(1, 769))]])[:, 1])
        recall_proba = round(recall_proba, 2)
    else:
        recall_proba = 1

    print(model_pass['model_name_or_number'])

    complaints = []
    for i in range(len(model_pass)):
        complaints.append(dict(
            prod_desc=model_pass.iloc[i, :].loc['product_description'], 
            manufacturer=model_pass.iloc[i, :].loc['manufacturer_/_importer_/_private_labeler_name'],  
            brand=model_pass.iloc[i, :].loc['brand'], 
            model_num=model_pass.iloc[i, :].loc['model_name_or_number'],
            incident_desc=model_pass.iloc[i, :].loc['incident_description']))

    
    return({'num_complaints':len(model_pass), 'recall_proba':recall_proba, 'complaints':complaints, 'brand':brand, 'product':product, 'model':model})
