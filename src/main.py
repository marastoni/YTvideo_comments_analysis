from googleapiclient.discovery import build
from keras.models import load_model
from keras.preprocessing import text
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
from os import environ as env
from textblob import TextBlob


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
    return commentList
    
    
    
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
    main(url)


