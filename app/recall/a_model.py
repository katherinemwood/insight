import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk, datetime, re, warnings
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from scipy import interp

def ModelIt(fromUser  = 'Default', raw_data = []):
  #Clean up the input
	if len(raw_data) == 0:
		print('ERROR')
		return

	cList = {
	  "ain't": "am not",
	  "aren't": "are not",
	  "can't": "cannot",
	  "can't've": "cannot have",
	  "'cause": "because",
	  "could've": "could have",
	  "couldn't": "could not",
	  "couldn't've": "could not have",
	  "didn't": "did not",
	  "doesn't": "does not",
	  "don't": "do not",
	  "hadn't": "had not",
	  "hadn't've": "had not have",
	  "hasn't": "has not",
	  "haven't": "have not",
	  "he'd": "he would",
	  "he'd've": "he would have",
	  "he'll": "he will",
	  "he'll've": "he will have",
	  "he's": "he is",
	  "how'd": "how did",
	  "how'd'y": "how do you",
	  "how'll": "how will",
	  "how's": "how is",
	  "I'd": "I would",
	  "I'd've": "I would have",
	  "I'll": "I will",
	  "I'll've": "I will have",
	  "I'm": "I am",
	  "I've": "I have",
	  "isn't": "is not",
	  "it'd": "it had",
	  "it'd've": "it would have",
	  "it'll": "it will",
	  "it'll've": "it will have",
	  "it's": "it is",
	  "let's": "let us",
	  "ma'am": "madam",
	  "mayn't": "may not",
	  "might've": "might have",
	  "mightn't": "might not",
	  "mightn't've": "might not have",
	  "must've": "must have",
	  "mustn't": "must not",
	  "mustn't've": "must not have",
	  "needn't": "need not",
	  "needn't've": "need not have",
	  "o'clock": "of the clock",
	  "oughtn't": "ought not",
	  "oughtn't've": "ought not have",
	  "shan't": "shall not",
	  "sha'n't": "shall not",
	  "shan't've": "shall not have",
	  "she'd": "she would",
	  "she'd've": "she would have",
	  "she'll": "she will",
	  "she'll've": "she will have",
	  "she's": "she is",
	  "should've": "should have",
	  "shouldn't": "should not",
	  "shouldn't've": "should not have",
	  "so've": "so have",
	  "so's": "so is",
	  "that'd": "that would",
	  "that'd've": "that would have",
	  "that's": "that is",
	  "there'd": "there had",
	  "there'd've": "there would have",
	  "there's": "there is",
	  "they'd": "they would",
	  "they'd've": "they would have",
	  "they'll": "they will",
	  "they'll've": "they will have",
	  "they're": "they are",
	  "they've": "they have",
	  "to've": "to have",
	  "wasn't": "was not",
	  "we'd": "we had",
	  "we'd've": "we would have",
	  "we'll": "we will",
	  "we'll've": "we will have",
	  "we're": "we are",
	  "we've": "we have",
	  "weren't": "were not",
	  "what'll": "what will",
	  "what'll've": "what will have",
	  "what're": "what are",
	  "what's": "what is",
	  "what've": "what have",
	  "when's": "when is",
	  "when've": "when have",
	  "where'd": "where did",
	  "where's": "where is",
	  "where've": "where have",
	  "who'll": "who will",
	  "who'll've": "who will have",
	  "who's": "who is",
	  "who've": "who have",
	  "why's": "why is",
	  "why've": "why have",
	  "will've": "will have",
	  "won't": "will not",
	  "won't've": "will not have",
	  "would've": "would have",
	  "wouldn't": "would not",
	  "wouldn't've": "would not have",
	  "y'all": "you all",
	  "y'alls": "you alls",
	  "y'all'd": "you all would",
	  "y'all'd've": "you all would have",
	  "y'all're": "you all are",
	  "y'all've": "you all have",
	  "you'd": "you had",
	  "you'd've": "you would have",
	  "you'll": "you you will",
	  "you'll've": "you you will have",
	  "you're": "you are",
	  "you've": "you have"
	}

	c_re = re.compile('(%s)' % '|'.join(cList.keys()))

	def expandContractions(text, c_re=c_re):
	    def replace(match):
	        return cList[match.group(0)]
	    return c_re.sub(replace, text)

	def clean_text(df, text_field):     
	    # taken from 'How to Solve 90% of NLP Problems' 
	    # remove links     
	    df[text_field] = df[text_field].str.replace(r"http\S+", "")
	    df[text_field] = df[text_field].str.replace(r"http", "")
	    #remove weird characters
	    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n\(\)]", " ")
	    df[text_field] = df[text_field].str.replace(r"@", "at")
	    df[text_field] = df[text_field].str.replace(r"amp", "and")

	    df[text_field] = df[text_field].str.lower()
	    df[text_field] = df[text_field].apply(lambda x: expandContractions(x))
	    df = df.fillna('')
	    return df

	def tv(data):
	    vectorizer = TfidfVectorizer()
	    emb = vectorizer.fit_transform(data)
	    return emb, vectorizer
	
	raw_data = raw_data[['product_description', 'brand', 'manufacturer_/_importer_/_private_labeler_name',
	                                                   'incident_description', 'label']]
	cleaned_text = clean_text(raw_data, 'incident_description')

	list_corpus = cleaned_text["incident_description"].tolist()
	list_labels = (cleaned_text["label"].values > 0).astype(int).tolist()

	tf_corpus, tfidf_vectorizer = tv(list_corpus)

	X_train_counts, X_test_counts, y_train, y_test = train_test_split(tf_corpus, list_labels, test_size=0.2, random_state=40)

	clf = LogisticRegression(C=1.0, penalty='l2',class_weight='balanced', solver='newton-cg', 
	                         multi_class='multinomial', n_jobs=-1, random_state=40)
	clf.fit(X_train_counts, y_train)

	y_pred = clf.predict(X_test_counts)

	f1 = sklearn.metrics.f1_score(y_test, y_pred)
	acc = sklearn.metrics.accuracy_score(y_test, y_pred)
	auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
	
	return({'F1':f1, 'Accuracy':acc, 'AUC':auc})