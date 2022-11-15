import pandas as pd
import joblib
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

query = "Is the Hurricane over or is it not over"

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tempTable', engine)

# # load model
# model = joblib.load("../models/classifier.pkl")

with open('../modelEvaluationDebug.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    model, X_test, Y_test, Y_pred = pickle.load(f)

accuracy = (Y_pred == Y_test).mean()
print(f"Accuracy = {accuracy}")

# select message to validate the pickling and loding of model is still giving similar results
messag_no = 0
# use pickle date to see what was predicted before pickling
classification_labels_pkl = Y_pred[0]
classification_results_pkl = dict(zip(df.columns[4:], classification_labels_pkl))
# use model to predict classification for query
classification_labels = model.predict([X_test[messag_no]])[0]
classification_results = dict(zip(df.columns[4:], classification_labels))
# accuracy
classification_results_pkl == classification_results

print(f"Message = {X_test[messag_no]}")
print("hi")