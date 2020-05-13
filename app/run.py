import json
import plotly
import pandas as pd
import re
from collections import OrderedDict

import nltk
nltk.download(['stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Pie
import plotly.express as px
from sklearn.externals import joblib
from sqlalchemy import create_engine
from keras.preprocessing.text import Tokenizer
from models.train_classifier import ProperNounExtractor, OfferExtractor


def tokenize(text):
    ''' 
    Custom tokenizer function:
        input - text, a string
        output - a list of strings that have been cleaned, lemmatized,
            with stopwords and URLs taken out 
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    # tokenize text
    words = word_tokenize(text.replace("'", ""))
    
    # initialise lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_word_ls = []
    for word in words:
        if not word in set(stopwords.words('english')):
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_word = lemmatizer.lemmatize(word.strip().lower())
            clean_word_ls.append(clean_word)
        
    return clean_word_ls


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # get word counts by using keras Tokenizer class
    tokenizer = Tokenizer(num_words = 200, 
                          filters='0123456789!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    # fit the tokenizer on cleaned messages to extract the counts
    tokenizer.fit_on_texts(df.clean_message)
    counts = tokenizer.word_counts
    # order word counts in descending order
    ordered_counts = OrderedDict(sorted(counts.items(), 
                                        key=lambda t: t[1], 
                                        reverse=True))
    count_list = [(key, ordered_counts[key]) for key in ordered_counts.keys()]
    
    # remove stopwords and take only 800 words with highest counts
    clean_word_counts = []
    for word,count in count_list[:800]:
        if not word in set(stopwords.words('english')):
            clean_word_counts.append((word, count))

    # turn word counts into pandas df to plot
    word_counts_df = pd.DataFrame(clean_word_counts, 
                                  columns = ['word','count'])  
    
    # extract data needed for visuals
    genres_df = df.groupby('genre')[['message']].count().reset_index()
    
    
    # Create first figure, a bar chart containing the highest word counts
    # in all the messages, to see what the most common words are
    fig1 = px.bar(word_counts_df[:22].iloc[::-1], x = 'count', y='word',
                 orientation='h', text = 'word')
    fig1.update_traces(texttemplate='%{text}', textposition='outside')
    fig1.update_traces(marker_color='#e1341e')
    fig1.update_layout(
        title_text = "Most frequent words in messages",
        xaxis=dict(
            showticklabels=True,
            title="Counts"
        ),
        yaxis=dict(
            showticklabels=False,
            title = "Word"
        )
    )
    
    # create second figure, a Pie chart of the genre breakdown in data
    fig2 = {'data': [
                Pie(
                    values = genres_df['message'],
                    labels = genres_df['genre'],
                    textinfo='label+percent',
                    insidetextorientation='radial')
                ],
            'layout': {
                'title': 'Breakdown of message genres',
                'showlegend': False}
        }

    figures = [fig1,fig2]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()