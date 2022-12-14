import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


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


# load training data from sql database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tempTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Initial landing page containing metrics visualising the data used to train the disaster classication model
    :return:
    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    no_weather_related = df[(df['weather_related'] == True) | (df['floods'] == True) | (df['storm'] == True) |
                            (df['other_weather'] == True)].shape[0]
    no_not_weather_related = df.shape[0] - no_weather_related
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    ids=['Residential', 'Non-Residential', 'Utility'],
                    values=genre_counts
                )
            ]
        }
        ,
        {
            'layout': {
                'title': 'Weather related messages'
            },
            'data': [
                Pie(
                    labels=['Weather related', 'Non-Weather related'],
                    ids=['Weather related', 'Non-Weather related'],
                    values=[no_weather_related, no_not_weather_related]
                )
            ]
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """
    # web page that handles user query and displays model results
    :return:
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Main code entry point. Starts web app.
    - The web app allows users to look at statistics of the training data used to train the disaster model.
    - The app allows the used to query the model with messages which are then classified using the pretrained model.
    :return:
    """
    app.run(host='0.0.0.0', port=3100, debug=True)


if __name__ == '__main__':
    main()
