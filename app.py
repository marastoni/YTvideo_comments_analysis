import streamlit as st
import pandas as pd
import plotly.express as px
import src.main 
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import csv

#set GIT_PYTHON_REFRESH=quiet


st.title('Youtube comments analysis')
st.sidebar.title('Youtube comments analysis')

st.markdown("This application is a streamlit dashboard to analize the youtube video's comments")
st.sidebar.markdown("This application is a streamlit dashboard to analize the youtube video's comments")

url = st.sidebar.text_input('Video Url:')
btn_load_data = st.sidebar.checkbox("Load the video's data")


def download_link_csv(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False,sep=';',quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def download_link_json(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_json()
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'





if btn_load_data:
    df = main.main(url)
    st.markdown('%i total comments ' % len(df))
    st.write(df)
    tmp_download_link_csv = download_link_csv(df, 'YOUR_DF.csv', 'Click here to download your data as CSV!')
    tmp_download_link_json = download_link_json(df, 'YOUR_DF.json', 'Click here to download your data as JSON!')
    col1,col2 = st.beta_columns(2)
    col1.markdown(tmp_download_link_csv, unsafe_allow_html=True)
    col2.markdown(tmp_download_link_json, unsafe_allow_html=True)
    
    st.sidebar.subheader('Show random comments:')
    utility = st.sidebar.radio('utility',('usefull','other'))
    sentiment = st.sidebar.radio('sentiment',('positive','neutral','negative'))
    random_comment = df.query('labels_utility == @utility & labels_sentiment == @sentiment').sample(n=1).iat[0,0]
    st.sidebar.markdown(random_comment)
  
    st.sidebar.subheader('Number of comments:')
    select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie chart'], key=1)
    # count = df['comments'].value_counts()
    # st.write(count)
    # count = pd.DataFrame({'comments':count.index, 'value':count.values})
    if not st.sidebar.checkbox('Hide',True):
        if select == 'Histogram':
            fig = px.bar(df,x = 'labels_sentiment',y =df['comments'], color = 'labels_utility',barmode='group',height = 500,labels={
                     "labels_sentiment": "labels_sentiment",
                     "y": "comments",
                     "labels_utility": "labels_utility"
                     })
            st.plotly_chart(fig)
        if select == 'Pie chart':
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
            fig.add_trace(go.Pie(labels=df['labels_utility'], values=df['comments'].value_counts(), name="utility"),
              1, 1)
            fig.add_trace(go.Pie(labels=df['labels_sentiment'], values=df['comments'].value_counts(), name="sentiment"),
              1, 2)
            fig.update_traces(hole=.4, hoverinfo="label+percent+name")
            fig.update_layout(
                        title_text='Number of comments',
                        # Add annotations in the center of the donut pies.
                        annotations=[dict(text='utility', x=0.18, y=0.5, font_size=16, showarrow=False),
                                    dict(text='sentiment', x=0.86, y=0.5, font_size=16, showarrow=False)])          
            st.plotly_chart(fig)















