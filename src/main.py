from googleapiclient.discovery import build
from keras.models import load_model
from keras.preprocessing import text
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
from os import environ as env
from textblob import TextBlob
from gensim.summarization.summarizer import summarize,summarize_corpus
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from sklearn.metrics import silhouette_score
from cleantext import clean


def Parse(response):
    # print('response =')
    comments_num= len(response['items'])
    # print('Number of comments =', comments_num)
    #topComment
    commentList=[]
    for i in range(comments_num):
        comment = response['items']
        comment_text = [comment[i]['snippet']['topLevelComment']['snippet']['textOriginal']][0]
        commentList.append(comment_text)
        # print('-'*10)
        # print(f'\t>>Comment:>>\n',comment_text)
        try:
            #Replies
            rep = comment[i]['replies']['comments']
            repLen = len(rep)
            for j in range(repLen):
                rep_text = [rep[j]['snippet']['textOriginal']][0]
                commentList.append(rep_text)
                # print('.'*10)
                # print(f'\t>>Replies:>>\n',rep_text)
        except:
            pass
                   
    return commentList
    

def PredSample(commentList):
    #utility anlysis
    model = load_model('model.h5')
    x_test_df = pd.Series(commentList,name='comments')
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(x_test_df)
    x_test = tokenizer.texts_to_sequences(x_test_df)
    maxlen = model.layers[0].input_shape[1]
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    pred = (model.predict(x_test) > 0.5).astype("int32")
    pred = np.where(pred==0, 'other', pred)
    pred = np.where(pred=='1', 'usefull', pred) 
    pred = pred.reshape([len(pred)])
    pred_df = pd.Series(pred,name='labels_utility')
    df = pd.concat([x_test_df,pred_df],axis=1)
    
    #sentiment analysis
    sentimentList=[]
    for sentence in x_test_df:
        blob = TextBlob(sentence)
        sentimentList.append(blob.polarity)
    for i in range(len(sentimentList)):
        if sentimentList[i] > 0:
            sentimentList[i] = 'positive'
        elif sentimentList[i] == 0:
            sentimentList[i] = 'neutral'
        else:
            sentimentList[i] = 'negative'
    df['labels_sentiment'] = sentimentList          

    return df
    
def clean_text(commentList):
    for i in range(len(commentList)):
        if '"' in commentList[i]:
            commentList[i] = commentList[i].replace('"','')
        if ';' in commentList[i]:
            commentList[i] = commentList[i].replace(';','')
        # commentList[i] = clean(commentList[i],
        #         fix_unicode=True,               # fix various unicode errors
        #         to_ascii=True,                  # transliterate to closest ASCII representation
        #         lower=True,                     # lowercase text
        #         no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        #         no_urls=True,                  # replace all URLs with a special token
        #         no_emails=True,                # replace all email addresses with a special token
        #         no_phone_numbers=True,         # replace all phone numbers with a special token
        #         no_numbers=True,               # replace all numbers with a special token
        #         no_digits=True,                # replace all digits with a special token
        #         no_currency_symbols=True,      # replace all currency symbols with a special token
        #         no_punct=True,                 # remove punctuations
        #         replace_with_punct="",          # instead of removing punctuations you may replace them
        #         replace_with_url="<URL>",
        #         replace_with_email="<EMAIL>",
        #         replace_with_phone_number="<PHONE>",
        #         replace_with_number="<NUMBER>",
        #         replace_with_digit="0",
        #         replace_with_currency_symbol="<CUR>",
        #         lang="en"                       # set to 'de' for German special handling
        #         )
    return commentList
    
def Summarization(df):
    commentList = df.to_list()
    corpus = '\n'.join(commentList)
    summary = summarize(corpus,word_count=100)
    return summary
   
def tokenizer(keyword):
    stemmer = PorterStemmer()
    # with open(,'r') as f:
    # stopwords = 
    return [stemmer.stem(w) for w in keyword.split(' ')]

       
def Clustering(df):
    commentList = df['comments'].to_list()
    tfidf = TfidfVectorizer(tokenizer=tokenizer)
    X = pd.DataFrame(tfidf.fit_transform(commentList).toarray(),
                     index=commentList,columns=tfidf.get_feature_names())
    opt_number = Auto_number_cluster(X)
    c = cluster.KMeans(n_clusters=opt_number, init='k-means++', max_iter=100, n_init=1)
    clu = c.fit_predict(X)
    df['group'] = pd.Series(clu)

    return df


def Auto_number_cluster(X):
    sil = []
    kmax = 8
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
      kmeans = cluster.KMeans(n_clusters = k).fit(X)
      labels = kmeans.labels_
      sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    
    opt_cluster_number = sil.index(max(sil))+2
    
    return opt_cluster_number
    
def main(url):
    videoID = url.split('=')[1]
    # with open(r'.\secret\private_api_key.txt') as f:
    #     lines = f.readlines()
    # for line in lines:
    #     if 'api' in line:
    #         api_key = line.split(':')[1]
    api_key = env.get('GOOGLE_API_KEY')
    youtube = build('youtube', 'v3', developerKey=api_key)
    usefull = []
    misc =[]
    total =[]
    request = youtube.commentThreads().list(
                part= "snippet,replies",
                maxResults=1000,
                videoId=videoID)
    response = request.execute()
    commentList = Parse(response)
    commentList = clean_text(commentList)
    df = PredSample(commentList)
    while True:
        try:
            request = youtube.commentThreads().list_next(request,response)
            response = request.execute()
            commentList = Parse(response)
            df = df.append(PredSample(commentList),ignore_index=True)
        except:
            break
    
    return df
    

if __name__ == '__main__':
    url= 'https://www.youtube.com/watch?v=wXGa0bDvFWA'
    df = main(url)
    df = Clustering(df)
    n_topic = max(df['group'])+1
    for i in range(n_topic):
        grp = df.groupby('group')['comments']
        mini_df = grp.get_group(i)
        summary = Summarization(mini_df)
        print(summary)
        print('*************************')

