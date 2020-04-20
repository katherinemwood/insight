## PRecall: Predicting product recalls from consumer complaints

PRecall is an application that makes it easier to find consumer complaints and recall notices for products and makes predictions about whether a set of consumer complaints indicate that a product might be recalled.

This repo contains the files for the Streamlit app (streamlit_app/) and the IPython notebooks used to merge the recall and complaint data sets and train the predictive model. The data cleaning and model fitting were done with Python (pandas and scikit-learn), while the app is deployed with [Streamlit](https://www.streamlit.io/).

### Data Sources

PRecall relies on two data sets, both furnished by the [Consumer Product Safety Commission](https://www.cpsc.gov/) and accessed via their API. One dataset is the collection of all recall announcements up to January 2020, and the other is all consumer safety complaints up to January 2020.

### Data Cleaning and Preparation

1. The datasets are pulled via the API and saved to CSV files.

2. Both datasets are processed to extract as much information from the unstructured data fields as possible, in addition to the stuctured information. This applies particularly to the database of product recalls, where most of the information needed to identify a product is contained in the unstructured text of the recall announcement.

3. Consumer complaints are paired up to the relevant recall announcement, if it exists, with a series of filtering operations. Complaints and recalls are first matched on likely brands, then on likely products, and finally by searching for common specifiers such as the model number or serial number between the complaint and remaining recall candidates. Complaints for which no recall can be found are labeled as "not recalled."

4. Using the NLTK library for Python, the text of the complaints is cleaned and TF-IDF encoded.

5. Using cross-valdation, the smoothing parameter for the L2 regularizer of a logistic regression classifier is selected. Each training fold is upsampled via SMOTE and validated against the non-upsampled validation fold.

6. The model is trained on the full, upsampled training set and tested on hold-out.

7. The model is trained on all upsampled data and pickled.

### Application

Users can search for a product type, brand name, and/or model name or number. Relevant products are found in the complaints database using a similar procedure to find the recalls. If these products have been recalled, the recall notice is fetched from the recall database. Otherwise, the complaints are run through the model to get a prediction of the likelihood of recall, which is displayed to the user along with the associated complaints.

### File Structure
`fetch_recall_data.ipynb`: This notebook pulls down the complaint and recall data, formats it, and saves it to .csv.  
`label_data.ipynb`: This notebook extracts information from unstructured fields and performs the matching between complaints and recalls, then saves the labeled data.  
`model.ipynb`: This notebook encodes the text from the complaints and trains the logistic regression model, saving out the final result.
`streamlit_app`: This directory contains the .py script that runs the app, the recall data, the complaint data, the encoded complaint data, and the pickled model.
