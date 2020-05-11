import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
# import libraries
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class OfferExtractor(BaseEstimator, TransformerMixin):

    def contains_offer(self, text):
        options = ['I can', 'I have', 'we have','we can', 'donate', 'give']
        return any([s in text for s in options])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.contains_offer)
        return pd.DataFrame(X_tagged)



def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    # fix category in 'related' column as want only 0 and 1 options
    df.loc[df.related == 2, 'related'] = 1
    X = np.asarray(df.message)
    Y = df.drop(columns = ['id','message','original','genre'])
    # extract the names of classification categories
    category_names = list(Y.columns)
    return X,Y,category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    # tokenize text
    words = word_tokenize(text)
    
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


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()),
            ('offer', OfferExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    params = { 'clf__estimator__min_samples_split': 3,
        'clf__estimator__n_estimators': 100,
        'features__text_pipeline__vect__max_df': 1.0,
        'features__text_pipeline__vect__max_features': 10000}
    
    pipeline.set_params(**params)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i,category in enumerate(category_names):
        f1 = f1_score(Y_test.iloc[:,i], Y_pred[:,i])
        precision = precision_score(Y_test.iloc[:,i], Y_pred[:,i])
        recall = recall_score(Y_test.iloc[:,i], Y_pred[:,i])
        print(category + ': ', 'Precision: ', precision, 'Recall: ', 
              recall, 'f1-score: ', f1)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()