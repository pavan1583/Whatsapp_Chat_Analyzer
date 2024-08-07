import re
import pandas as pd
import numpy as np
import emoji
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Extract the Date time
def date_time(s):
    pattern = r'\d{1,2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s?[ap]m\s-\s'
    result = re.match(pattern, s)
    return bool(result)


# Extract contacts
def find_contact(s):
    s = s.split(":")
    return len(s) == 2


# Extract Message
def get_message(line):
    split_line = line.split(' - ')
    datetime = split_line[0]
    date, time = datetime.split(', ')
    message = " ".join(split_line[1:])

    if find_contact(message):
        split_message = message.split(": ", 1)
        author = split_message[0]
        message = split_message[1]
    else:
        author = None

    return date, time, author, message


# Initialize data list
data = []

# Read the WhatsApp chat text file
conversation = 'WhatsApp Chat with FEAT_B.Tech._All_2024-25.txt'
with open(conversation, encoding="utf-8") as fp:
    fp.readline()  # Skip the first line
    message_buffer = []
    date, time, author = None, None, None

    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if date_time(line):
            if len(message_buffer) > 0:
                data.append([date, time, author, ''.join(message_buffer)])
            message_buffer.clear()
            date, time, author, message = get_message(line)
            message_buffer.append(message)
        else:
            message_buffer.append(line)

# Create DataFrame
df = pd.DataFrame(data, columns=["Date", "Time", "Contact", "Message"])
df['Date'] = pd.to_datetime(df['Date'])

# Drop rows with NaN values
data = df.dropna()

# Sentiment analysis
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Message"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Message"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Message"]]

# Display a sample of the data
print(data.sample(50))

