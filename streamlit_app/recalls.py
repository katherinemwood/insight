import streamlit as st
import pandas as pd
import numpy as np
import os, pickle, sklearn, re, scipy
from nltk.stem.snowball import SnowballStemmer
from fuzzywuzzy import fuzz, process
from sklearn.linear_model import LogisticRegression

#Given an integer probability between 0 and 100,
#return a string label for the likelihood.
def label_probability(prob):
    if prob >= 0 and prob <= 24:
        return('very unlikely')
    elif prob >= 25 and prob <= 49:
        return ('somewhat unlikely')
    elif prob >= 50 and prob <= 74:
        return ('more likely than not')
    elif prob >= 50 and prob <= 74:
        return ('likely')

#Compare a reference string to each member of a set of comparison strings by looking for
#matching words or performing a fuzzy match on the tokens.
#The threshold for how closely the strings must match for fuzzy matching can be set.
#Match requriements can be "fuzzy," for token matches only, "word" if a common word must
#exist between the reference and comparison strings, or "both" if both must be true for
#a match to be declared.
def fuzzy_match(reference_strings, comparison_strings, threshold=80, match_requirements='both'):
    #Fuzzy string match
    if isinstance(comparison_strings, str):
        comparison_strings = comparison_strings.strip('][').replace("'", '').split(', ')
    if not isinstance(comparison_strings, list):
        return 0
    comparison_strings = [string.lower() for string in comparison_strings]
    if not comparison_strings:
        return 0
    fuzzy_match = (pd.Series(list(zip(*process.extract(' '.join(reference_strings), comparison_strings,
                                                       limit=len(comparison_strings),
                                                     scorer = fuzz.token_set_ratio)))[1]) > threshold).any()
    #Whole-word match
    comp_words = set(comparison_strings)
    ref_words = set(reference_strings)
    common_word_match = len(ref_words.intersection(comp_words)) > 0
    if match_requirements == 'both':
        return fuzzy_match and common_word_match
    elif match_requirements == 'fuzzy':
        return fuzzy_match
    elif match_requirements == 'word':
        return common_word_match
    else:
        return 0

@st.cache
#Given a reference string, a set of data to search through, a column with the relevant field,
#a match threshold, and match requriements, return a subset of the data where the string was
#found in the indicated column to the given requirements.
def matches_on_field(reference_string, search_set, comparison_column, threshold = 80, match_reqs='both'):
    if isinstance(reference_string, pd.core.series.Series):
        reference_string = reference_string.values[0]
    if pd.isnull(reference_string) or not reference_string:
        return([])
    else:
        mask = search_set[comparison_column].apply(lambda comp_string: fuzzy_match(reference_string.split(' '), 
                                                                             comp_string, threshold, 
                                                                                   match_reqs)).astype(bool)
        matches = search_set[mask.values]                                     
        return matches

#Given a string and a stemmer, strip out non-alphanumeric characters,
#remove extra whitespace, lower the case, split the string into tokens,
#and stem each token. If FUSE is true, put the list back into a single
#whitespace-separated string. Otherwise, return a list of clean tokens.
def clean_input(str_input, stemmer, fuse=True):
    pattern = re.compile('[^a-z0-9]')
    tokens = []
    tokens = str_input.strip(' ').split(' ')
    cleaned_tokens = []
    for token in tokens:
        token = token.lower()
        if not re.match(pattern, token):
            token = pattern.sub('', token)
            token = stemmer.stem(token)
            cleaned_tokens.append(token)
    return ' '.join(list(set(cleaned_tokens))) if fuse else list(set(cleaned_tokens))

#Given data to search through, a product, and a brand, return related recalls.
def find_matching_recalls(data, product_input, brand_input):
    stemmer = SnowballStemmer("english")
    data.loc[:, ('Title')] = data.loc[:, ('Title')].fillna('')
    data.loc[:, ('clean_title')] = data.loc[:, ('Title')].apply(clean_input, stemmer=stemmer, fuse=False)
    product = clean_input(product_input, stemmer) if product_input else ''
    brand = clean_input(brand_input, stemmer) if brand_input else ''
    product_pass = matches_on_field(product, data, 'clean_title') if product else data
    brand_pass = matches_on_field(brand, product_pass, 'clean_title') if brand else product_pass
    return brand_pass

#Given a set of data, inputs for the product type, brand, and model number, return the related
#products. Track and report progress with a given progress bar and status message, as this process
#can take a while.             
def find_matching_products(data, product_input, brand_input, model_input, prog_bar, status_message):
    stemmer = SnowballStemmer("english")
    data.loc[:, ('Model Name or Number')] = data.loc[:, ('Model Name or Number')].fillna('')
    data.loc[:, ('clean_model')] = data.loc[:, ('Model Name or Number')].apply(lambda model_string: model_string.strip(' ').split(' '))
    data.loc[:, ('brand')] = data.loc[:, ('Brand')].fillna('')
    data.loc[:, ('brand')] = data.loc[:, ('brand')].apply(clean_input, stemmer=stemmer)
    data.loc[:, ('clean_brand')] = data.loc[:, ('clean_brand')].apply(lambda brand_str: brand_str.strip('][').replace("'", '').split(', '))
    data.loc[:, ('brand')] = data.apply(lambda row: row.loc[('clean_brand')] + row.loc[('brand')].split(' '), axis=1)
    product = clean_input(product_input, stemmer) if product_input else ''
    brand = clean_input(brand_input, stemmer) if brand_input else ''
    model = model_input.strip(' ').lower()
    prog_bar.progress(5)
    status_message.text('Searching for related products...')
    product_pass = matches_on_field(product, data, 'clean_product') if product else data
    prog_bar.progress(45)
    status_message.text('Searching for related brands...')
    brand_pass = matches_on_field(brand, product_pass, 'brand') if brand else product_pass
    prog_bar.progress(85)
    status_message.text('Searching for related models...')
    model_pass = matches_on_field(model, brand_pass, 'clean_model') if model else brand_pass
    prog_bar.progress(100)
    status_message.text('Cleaning up...')
    return model_pass

#Given rows from the full recall dataset, return only the relevant columns
#for display and rename them nicely.
def format_recall(recalls):
    display_recall = recalls.loc[:, ('RecallDate', 'Title', 'Description', 'ConsumerContact')]
    display_recall.columns = ['Date', 'Title', 'Details', 'Contact']
    return display_recall

#Given rows from the full complaint dataset, return only the relevant columns
#for display and rename them nicely.
def format_complaints(complaints):
    display_complaints = complaints.loc[:, ('Product Description', 'Manufacturer / Importer / Private Labeler Name', 'Brand', 
            'Model Name or Number', 'Incident Description')]
    display_complaints.columns = ['Product Description', 'Company', 'Brand', 'Model', 'Complaint']
    return display_complaints

st.title('PRecall')

product_input = st.sidebar.text_input(label='What type of product is it?', value='', key='product_input')
brand_input = st.sidebar.text_input(label='What brand of product is it?', value='', key='brand_input')
model_input = st.sidebar.text_input(label='Do you have the model name or number?', value='', key='model_input')

prog = st.empty()
prog_text = st.empty()

inst = st.subheader('Search for a product, a brand, or a specific model number.')

#Once the button is clicked, run the search
if st.sidebar.button('Find Recall Information'):
    inst.empty()

    if not any([product_input, brand_input, model_input]):
        st.write("Please enter at least one search term.")
    else:
        prog.progress(0)
        prog_text.text('Seaching database...')
        raw_data = pd.read_csv('clean_labeled_data.csv', encoding="ISO-8859-1", dtype='object')
        embeddings = scipy.sparse.load_npz('tfidf_embeddings.npz')
        logreg_model = pickle.load(open('trained_model.sav', 'rb'))
        recalls = pd.read_csv('recalls.csv', encoding="ISO-8859-1", dtype='object')

        results = find_matching_products(raw_data, product_input, brand_input, model_input, prog, prog_text)

        recall_proba = 0
        assoc_recall = []

        if len(results) == 0:
            display_complaints = 'Nothing to show.'
        elif not model_input:
            recall_text = "To get the likelihood of a specific product being recalled, please enter a model name or number."
            recalls = find_matching_recalls(recalls, product_input, brand_input)
            assoc_recall = format_recall(recalls)
            display_complaints = format_complaints(results).sort_values(by='Company')
        else:
            if (results['labels'].values.astype(int) > 0).sum() == len(results) and len(set(results['labels'].values)) == 1:
                recall_proba = 1
                assoc_recall = format_recall(recalls[recalls['RecallID'] == results.iloc[0].loc['labels']])
                recall_text = '<span style="color:red">**This product has been recalled.**</span>'
            elif not (results['labels'].values.astype(int) > 0).any():
                recall_proba = np.mean(logreg_model.predict_proba(embeddings[results.index, :])[:, 1])
                recall_text = 'The model estimates the chance of recall at {0:.0f}% for this product. This product is **{1:s}** to be recalled.'.format(recall_proba*100, label_probability(recall_proba*100))
            else:
                pred_proba = logreg_model.predict_proba(embeddings[results.index, :])[:, 1]
                weighted_proba_v = ((pred_proba + np.array(results['labels'].values.astype(int) > 0, dtype = int)) 
                    / (1 + np.array(results['labels'].values.astype(int) > 0, dtype = int)))
                weighted_proba = np.mean(weighted_proba_v)

                labeled_recalls = results.loc[results['labels'].values.astype(int) > 0, ('labels')]

                assoc_recall = format_recall(recalls[recalls['RecallID'].apply(lambda recallid: recallid in set(labeled_recalls))])
                recall_text = "This is a tricky one. This product may be associated with these recalls." + \
                " The model is pretty sure it's likely to be recalled if it hasn't been already, at {0:.0f}% chance.".format(weighted_proba*100)

            display_complaints = format_complaints(results)

        #Display the results
        prog.empty()
        prog_text.empty()
        st.subheader('Search Results')
        st.write('Your search yielded %d related complaints.' % len(results))
        if len(results) > 0:
            st.subheader('Recall Information')
            st.markdown(recall_text, unsafe_allow_html=True)
        if not isinstance(assoc_recall, list):
            st.subheader('Associated Recalls')
            st.table(assoc_recall)

        if not isinstance(display_complaints, str):
            st.subheader('Associated Complaints')
            st.table(display_complaints)

