import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer 
import string 
import nltk # NLP lai
from nltk.corpus import stopwords 
import fitz #read pdf content /convert pdf to image etc
import pickle 

# downloaded stopped words and initialized vectorizer 
nltk.download('stopwords')
vectorizer = CountVectorizer()

def pre_process_df():
    f_df = pd.DataFrame(columns=['Text','Label']) #inittialized df in the form of text and label
    df = pd.read_csv('Dataset.csv') #load csv
    f_df['Text'] = df['Text'] 
    f_df['Label'] = df['Label']
    return f_df #returned text and label

def input_process(text):
    translator = str.maketrans('','',string.punctuation)
    nopunc = text.translate(translator)
    words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')] #list comprehension technique
    return ' '.join(words)

def remove_stop_words(ip):
    final_ip = [] 
    for line in ip:
        line = input_process(line)
        final_ip.append(line)
    return final_ip



def train_model(df):
    X = df['Text']
    y = df['Label']
    X = remove_stop_words(X)
    df['Text'] = X
    X = vectorizer.fit_transform(X)
    nb = MultinomialNB()
    nb.fit(X,y)
    return nb



if __name__ == "__main__":
    df = pre_process_df()
    model = train_model(df)
    pickle.dump(model,open('classifier.model','wb'))
    pickle.dump(vectorizer,open('vectorizer.pickle','wb'))