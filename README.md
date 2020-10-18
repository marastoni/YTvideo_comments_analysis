# YTvideo_comments_analysis
The app is hosted by heroku server:
https://youtubevideocommentanalyzer.herokuapp.com/

Just put the YouTube video url and the script:
- classify the comment in 2 categories 'usefull' and 'misc'
- do a simple sentiment analysis classifing the comments into 'positive', 'neutral', 'negative'

# FIRST STEP

Use ``pip install -r requirements.txt`` to install the libraries needed.

Libraries that are needed to install:

- streamlit
- plotly
- keras
- google-api-python-client
- tensorflow-cpu
- textblob
- nltk

# SECOND STEP
Is need to set your own GOOGLE API KEY as environment variable called 'GOOGLE_API_KEY'.


# THIRD STEP
Run the app from command prompt in the project folder with ``streamlit run app.py``