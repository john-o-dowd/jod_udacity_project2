import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# https://scikit-learn.org/stable/modules/generated/
# sklearn.multioutput.MultiOutputClassifier.html#sklearn-multioutput-multioutputclassifier
def load_data(database_filepath):
    """
    Load a message data from database. Select a subset of the data (selected categories) and return them data
    :param database_filepath:
    :return: messages, message dataframe for selected categories, names of selected categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('tempTable', engine)
    X = df.message.values
    # names = ["aid_related", "medical_help"]
    names = df.columns[5:]
    Y = df[names]
    return X, Y, names


def tokenize(text):
    """
    Tokenizes text and returns cleaned version of tokens
    :param text:
    :return: clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    create pipeline model to categorize messages
    :return: model pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__n_jobs': [8]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Genenerate statistics on the performance of the model based on its ability to categorize the test data
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return: None
    """
    Y_pred = model.predict(X_test)
    # accuracy = (Y_pred == Y_test).mean()
    # print(f"Accuracy = {accuracy}")
    # calculate and print out performance metrics for model
    for ix, col in enumerate(Y_test.columns):
        f1 = f1_score(Y_test[col].values, Y_pred[:, ix], average='macro')
        recall = recall_score(Y_test[col].values, Y_pred[:, ix], average='macro')
        accuracy_col = (Y_test[col].values == Y_pred[:, ix]).mean()
        precision = precision_score(Y_test[col].values, Y_pred[:, ix], average='macro')
        print(f"Category : {col}, f1 = {f1:.2f}, recall = {recall:.2f}, accuracy = {accuracy_col:.2f}, precision = {precision:.2f}")

    # save all the prediction statistics for debug
    debug = False

    # if in debug save model and data used to test and predict it for additional model analysis
    if debug:
        with open('modelEvaluationDebug.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([model, X_test, Y_test, Y_pred], f)


def save_model(model, model_filepath):
    """
    store model in pickle file. This file is then loaded by the web app and used to categorize
    user messages entered through the web interface
    :param model:
    :param model_filepath:
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))  # save model in pickle file. This model is then loaded by we app


def main():
    """
    Main code entry poing:
    Build and save classifer model to categorize disaster messages. The model is trained based on user
    supplied pre-categorized training messages
    :return: None
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
