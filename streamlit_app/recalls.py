import streamlit as st
import pandas as pd
import numpy as np
import os, pickle, sklearn, psycopg2
from fuzzywuzzy import fuzz, process
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

@st.cache
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

user = 'katherinewood' #add your username here (same as previous postgreSQL)                      
host = 'localhost'
dbname = 'insight'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

st.title('PRecall')

product_input = st.sidebar.text_input(label='What type of product is it?', value='', key='product_input')
brand_input = st.sidebar.text_input(label='What brand of product is it?', value='', key='brand_input')
model_input = st.sidebar.text_input(label='Do you have the model name or number?', value='', key='model_input')

prog = st.empty()
prog_text = st.empty()

inst = st.subheader('Search for a product, a brand, or a specific model number.')

if st.sidebar.button('Find Recall Information'):

    inst.empty()
    
    prog.progress(0)
    prog_text.text('Seaching database...')

    query = "SELECT * FROM labeled_data"
    query_results = pd.read_sql_query(query,con)
    embeddings = np.concatenate([np.load('embedding_chunks/'+chunk) for chunk in os.listdir('embedding_chunks')], axis = 0)
    embedded_data = pd.concat([query_results.iloc[range(len(embeddings)), :], pd.DataFrame(embeddings)], axis=1)
    embedded_data.columns = embedded_data.columns.astype(str)

    prog.progress(5)
    prog_text.text('Searching for related products...')
    product_pass = find_search_match(embedded_data, product_input, 'clean_product') if product_input else embedded_data
    prog.progress(33)
    prog_text.text('Searching for related brands...')
    brand_pass = find_search_match(product_pass, brand_input, 'clean_brand') if brand_input else product_pass
    prog.progress(66)
    prog_text.text('Searching for related models...')
    model_pass = find_search_match(brand_pass, model_input, 'model_name_or_number', 100) if model_input else brand_pass
    prog.progress(100)

    logreg_model = pickle.load(open('trained_model.sav', 'rb'))

    recall_proba = 0

    if len(model_pass) == 0:
        display_complaints = 'Nothing to show.'
    else:
        if (model_pass['0'].values > 0).sum() < len(model_pass):
            recall_proba = np.mean(logreg_model.predict_proba(model_pass[[str(i) for i in list(range(1, 769))]])[:, 1])
        else:
            recall_proba = 1

        display_complaints = model_pass.loc[:, ('product_description',   'manufacturer_/_importer_/_private_labeler_name', 'brand', 
        'model_name_or_number', 'incident_description')]
        display_complaints.columns = ['Product Description', 'Company', 'Brand', 'Model', 'Complaint']

    prog.empty()
    prog_text.empty()

    st.subheader('Search Results')
    st.write('Your search yielded %d related complaints.' % len(model_pass))
    st.write('The model estimates the chance of recall at {0:.0f}% for this product.'.format(recall_proba*100))

    st.subheader('Associated Complaints')
    st.table(display_complaints)

