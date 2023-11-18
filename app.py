# Import necessary libraries


import requests
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import nltk
import langid
plt.style.use('fivethirtyeight')
import streamlit as st

from ntscraper import Nitter
scraper = Nitter()

##For Mail

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Set the page configuration
st.set_page_config(layout="wide")

# Set the page title
st.title("Sentiment Analysis")
st.subheader("Analyze sentiment and display top comments from a verious platform")


#defining Veriables
positive_percentage=0;negative_percentage=0;neutral_percentage=0; score=0;
wiki_positive_percentage=0;wiki_negative_percentage=0;wiki_neutral_percentage=0; score2=0;
tweet_positive_percentage=0;tweet_negative_percentage=0;tweet_neutral_percentage=0; score3=0;
web_url='';wiki_url=''; term='';

# Function to clean the text
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text

# Function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Function to compute the negative, neutral, and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Create columns for layout
col1, col2 = st.columns([1, 1])

# User input for the website URL
with col1:
    st.subheader("Fetching Data from Website")
    web_url = st.text_input("Enter the website URL:")
    fetch_button = st.button("Fetch Top 5 Comments")
    if not web_url:
        st.warning("Please enter a website URL.")

# Fetch and display reviews
if web_url:
    with col2:
        try:
            r = requests.get(web_url)
            soup = BeautifulSoup(r.text, 'html.parser')
            regex = re.compile('.*comment.*')
            results = soup.find_all('p', {'class': regex})
            reviews = [result.text for result in results]

            df = pd.DataFrame({'Comments': reviews})

            # Cleaning text
            df['Comments'] = df['Comments'].apply(cleanTxt)

            # Create two new columns
            df['Subjectivity'] = df['Comments'].apply(getSubjectivity)
            df['Polarity'] = df['Comments'].apply(getPolarity)

            df['Analysis'] = df['Polarity'].apply(getAnalysis)

            # Display sentiment analysis results
            st.write("### Sentiment Analysis Results:")
            st.write("Number of Comments:", df.shape[0])

            # Get the percentage of positive comments
            positive_percentage = round((df['Analysis'] == 'Positive').mean() * 100, 1)
            st.write(f"Percentage of positive comments: {positive_percentage}%")

            # Get the percentage of negative comments
            negative_percentage = round((df['Analysis'] == 'Negative').mean() * 100, 1)
            st.write(f"Percentage of negative comments: {negative_percentage}%")

            # Get the percentage of neutral comments
            neutral_percentage = round((df['Analysis'] == 'Neutral').mean() * 100, 1)
            st.write(f"Percentage of neutral sentiment: {neutral_percentage}%")

            # Final Score
            score = int((int(neutral_percentage / 2) + positive_percentage) / 10)
            st.write(f"So, on a scale of 1-10, we give it a score of: {score}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        # Show the value counts
        try:
            value_counts = df['Analysis'].value_counts()

            # Display the counts
            st.write("### Sentiment Analysis Counts:")
            st.write(value_counts)

            # Plot and visualize the counts
            st.bar_chart(value_counts)

        except:
            st.warning("No data available for sentiment analysis.")

if web_url and fetch_button:
    with col1:
        st.write("### Top 5 Comments:")
        st.table(df.head(5)[['Comments']].reset_index(drop=True))


# Create columns for layout
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Fetching Data from Wikipedia")
    wiki_url = st.text_input("Enter the Wikipedia URL:")
    fetch_button_wiki = st.button("Fetch Top 5 Comments", key="fetch_button_wiki")  # Add a unique key
    if not wiki_url:
        st.warning("Please enter a Wikipedia URL.")

if wiki_url:
    with col4:
        try:
            wiki = requests.get(wiki_url)
            wiki_comments = BeautifulSoup(wiki.text, 'html.parser')
            wiki_comments = wiki_comments.get_text(strip=True)

            wiki_comments = re.sub(r'\[\d+\]', "", wiki_comments)
            wiki_comments = re.sub(r'\([^)]*\)', '', wiki_comments)
            wiki_comments = re.sub(r'\[\w+\]', "", wiki_comments)
            wiki_comments = re.sub('[0-9]+', "", wiki_comments)
            cleanTxt(wiki_comments)

            def is_english(sentence):
                lang, _ = langid.classify(sentence)
                return lang == 'en'

            # Split the text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', wiki_comments)

            # Filter out non-English sentences
            english_sentences = [sentence.strip() for sentence in sentences if is_english(sentence)]

            # Join the English sentences back into a string
            english_text = ' '.join(english_sentences)

            sentence = sent_tokenize(english_text)
            # Create a dataframe with a column called comments
            df = pd.DataFrame({'Wiki Comments': sentence})
            # Download the file as Excel
            df.to_excel('wikipedia_file.xlsx', index=False)

            df['Subjectivity'] = df['Wiki Comments'].apply(getSubjectivity)
            df['Polarity'] = df['Wiki Comments'].apply(getPolarity)

            st.write("### Sentiment Analysis Results:")
            st.write("Number of Comments:", df.shape[0])

            df['Analysis'] = df['Polarity'].apply(getAnalysis)

            # Get the percentage of positive comments
            wiki_positive_percentage = round((df['Analysis'] == 'Positive').mean() * 100, 1)
            st.write(f"Percentage of positive comments: {wiki_positive_percentage}%")

            # Get the percentage of negative comments
            wiki_negative_percentage = round((df['Analysis'] == 'Negative').mean() * 100, 1)
            st.write(f"Percentage of negative comments: {wiki_negative_percentage}%")

            # Get the percentage of neutral comments
            wiki_neutral_percentage = round((df['Analysis'] == 'Neutral').mean() * 100, 1)
            st.write(f"Percentage of neutral sentiment: {wiki_neutral_percentage}%")

            # Final Score
            score2 = int((int(wiki_neutral_percentage / 2) + wiki_positive_percentage) / 10)
            st.write(f"So, on a scale of 1-10, we give it a score of: {score2}")


        except Exception as e:

            st.error(f"An error occurred: {e}")

        # Show the value counts

        try:
            value_counts = df['Analysis'].value_counts()

            # Display the counts

            st.write("### Sentiment Analysis Counts:")
            st.write(value_counts)

            # Plot and visualize the counts
            st.bar_chart(value_counts)
        except:

            st.warning("No data available for sentiment analysis.")

# Display the sentiment analysis counts
if wiki_url and fetch_button_wiki:
    with col3:
        st.write("### Top 5 Comments:")
        st.table(df.head(5)[['Wiki Comments']].reset_index(drop=True))



# Create columns for layout
col5, col6 = st.columns([1, 1])

with col5:
    st.subheader("Fetching Data from Twitter")
    # Get user input for search term and mode
    search_term = st.text_input("Enter the search term:")
    tweet_mode = st.selectbox("Select the mode:", ["term", "hashtag"])

    fetch_button_tweet = st.button("Fetch Top 5 Comments", key="fetch_button_tweet")  # Add a unique key

if fetch_button_tweet and tweet_mode:
    with col6:
        try:
            results = scraper.get_tweets(search_term, mode=tweet_mode, number=170)
            final_tweets = []

            for tweet in results['tweets']:
                data = [tweet['link'], tweet['text'], tweet['date'], tweet['stats']['likes'], tweet['stats']['comments']]
                final_tweets.append(data)
            data = pd.DataFrame(final_tweets, columns=['link', 'text', 'date', 'No_of_Likes', 'No_of_tweets'])

            # remove few colomns
            columns_to_remove = ['link', 'date', 'No_of_Likes', 'No_of_tweets']
            df_tweet = data.drop(columns=columns_to_remove)

            ##Cleaning all text
            # remove emojies
            def remove_emojis(text):
                emoji_pattern = re.compile("["
                                           u"\U0001F600-\U0001F64F"  # emoticons
                                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                           u"\U00002702-\U000027B0"  # Dingbats
                                           u"\U000024C2-\U0001F251"
                                           "]+", flags=re.UNICODE)
                return emoji_pattern.sub(r'', text)

            df_tweet['text'] = df_tweet['text'].apply(cleanTxt)
            df_tweet['text'] = df_tweet['text'].apply(remove_emojis)

            df_tweet['Subjectivity'] = df_tweet['text'].apply(getSubjectivity)
            df_tweet['Polarity'] = df_tweet['text'].apply(getPolarity)

            df_tweet['Analysis'] = df_tweet['Polarity'].apply(getAnalysis)

            # Get the percentage of positive comments
            tweet_positive_percentage = round((df_tweet['Analysis'] == 'Positive').mean() * 100, 1)
            st.write(f"Percentage of positive comments: {tweet_positive_percentage}%")

            # Get the percentage of negative comments
            tweet_negative_percentage = round((df_tweet['Analysis'] == 'Negative').mean() * 100, 1)
            st.write(f"Percentage of negative comments: {tweet_negative_percentage}%")

            # Get the percentage of neutral comments
            tweet_neutral_percentage = round((df_tweet['Analysis'] == 'Neutral').mean() * 100, 1)
            st.write(f"Percentage of neutral sentiment: {tweet_neutral_percentage}%")

            # Final Score
            score3 = int((int(tweet_neutral_percentage / 2) + tweet_positive_percentage) / 10)
            st.write(f"So, on a scale of 1-10, we give it a score of: {score3}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Show the value counts

        try:
            value_counts = df_tweet['Analysis'].value_counts()

            # Display the counts

            st.write("### Sentiment Analysis Counts:")
            st.write(value_counts)

            # Plot and visualize the counts
            st.bar_chart(value_counts)
        except:
            st.warning("No data available for sentiment analysis.")

if search_term and tweet_mode and fetch_button_tweet:
    with col5:
        st.write("### Top 5 Comments:")
        st.table(df_tweet.head(5)[['text']].reset_index(drop=True))




##Email Send:

smtp_port = 587  # Standard secure SMTP port
smtp_server = "smtp.gmail.com"  # Google SMTP Server

# Set up the email lists
email_from = "arindamg593@gmail.com"
email_list = []
st.subheader("Write Your Email:")
input_str = st.text_input("Enter email where you want to send data: ")
email_list.append(input_str)

# Define the password (better to reference externally)
pswd = "ofmk dssm pxgr qvod"  # As shown in the video this password is now dead, left in as example only

# name the email subject
subject = "Sentiment Analysis"


# Define the email function (dont call it email!)
def send_emails(email_list):
    for person in email_list:
        # Make the body of the email
        body = result_text = f"""
                By analyzing this ({web_url}), we found:
                {positive_percentage}% of users gave positive comments.
                {negative_percentage}% of users gave negative comments.
                {neutral_percentage}% of users gave neutral comments.
                So, on a scale of 1-10, we give it a score of {score}.

                You will find the entire comments file attached to this email as web_file.xlsx.


                By analyzing this wikipedia url({wiki_url}), we found:
                {wiki_positive_percentage}% of users gave positive comments.
                {wiki_negative_percentage}% of users gave negative comments.
                {wiki_neutral_percentage}% of users gave neutral comments.
                So, on a scale of 1-10, we give it a score of {score2}.

                You will find the entire comments file attached to this email as wikipedia_file.xlsx.


                By analyzing all the tweets of {search_term} we found:
                {tweet_positive_percentage}% of users gave positive comments.
                {tweet_negative_percentage}% of users gave negative comments.
                {tweet_neutral_percentage}% of users gave neutral comments.
                So, on a scale of 1-10, we give it a score of {score3}.

                You will find the entire comments file attached to this email as tweet.xlsx.
                """

        # make a MIME object to define parts of the email
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = person
        msg['Subject'] = subject

        # Attach the body of the message
        msg.attach(MIMEText(body, 'plain'))

        # Define the file to attach
        filename1 = "wikipedia_file.xlsx"
        filename2 = "web_file.xlsx"
        filename3 = "tweets.xlsx"

        # Open the file in python as a binary
        attachment1 = open(filename1, 'rb')  # r for read and b for binary
        attachment2 = open(filename2, 'rb')
        attachment3 = open(filename3, 'rb')

        # Encode as base 64 -1
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment1).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + filename1)
        msg.attach(attachment_package)

        # Encode as base 64 -2
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment2).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + filename2)
        msg.attach(attachment_package)

        # Encode as base 64 -3
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment3).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + filename3)
        msg.attach(attachment_package)

        # Cast as string
        text = msg.as_string()

        # Connect with the server
        st.write("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls()
        TIE_server.login(email_from, pswd)
        st.write("Succesfully connected to server\n")

        # Send emails to "person" as list is iterated
        st.write(f"Sending email to: {person}...")
        TIE_server.sendmail(email_from, person, text)
        st.write(f"Email sent to: {person}\n")

    # Close the port
    TIE_server.quit()


# Run the function
if(input_str):
    send_emails(email_list)