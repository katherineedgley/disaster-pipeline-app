import sys


# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re



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



def load_data(messages_filepath, categories_filepath):
    '''
    Function to read in CSV files and merge them
    Inputs both the filepath to CSV with message data and to CSV with
    categories data
    Outputs merged pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    Function that cleans and processes the dataframe with message
    and corresponding category data. Cleans and removes duplicates
    and adds column with tokenized messages for use in 
    '''
    # split each entry into a column
    categories = df.categories.str.split(';', expand=True)
    # extract column names and reset 
    category_colnames = categories.iloc[0].str[:-2]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # replace the categories column with the categories df we cleaned above
    df.drop(columns = ['categories'], inplace = True)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    
    # make new column with tokenized/cleaned messages
    df['clean_message'] = df.message.apply(lambda x: ' '.join(tokenize(x)))
    return df


def save_data(df, database_filename):
    '''
    Function to insert the cleaned dataframe into an SQL
    database, with the dataset titled 'messages'
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()