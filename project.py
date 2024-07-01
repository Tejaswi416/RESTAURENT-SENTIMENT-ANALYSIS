from google.colab import files
import io  # Import the io module
import pandas as pd
uploaded = files.upload()
data=pd.read_csv("Reviews.csv")
print(data)
data.head()
data.tail()
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
combined_text=" ".join(data['Review']) #Combine all review text into one string
wordcloud=WordCloud(width=800,height=400,background_color='purple').generate(combined_text)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()
from collections import Counter
targeted_words=['good','great','amazing','bad','not bad']
all_words=" ".join(data['Review']).lower().split()     #flattened reviews into a single list of words
word_counts=Counter(all_words)
target_word_count={word:word_counts[word] for word in targeted_words}
#plotting
plt.figure(figsize=(8,6))
plt.bar(target_word_count.keys(),target_word_count.values(),color=['blue','black','orange','red','black'])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Frequency of specific words in reviews')
plt.show()


lowercases_text = data['Review'].str.lower()
print(lowercases_text)
#tokenization
from nltk.tokenize import word_tokenize
import nltk

# Download the 'punkt' tokenizer
nltk.download('punkt')
data['Tokens'] = data['Review'].apply(word_tokenize)
print(data ['Tokens'] )
data.info()
#removing stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['Tokens'] = data['Review'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])
print(data['Tokens'])
#stemming
from nltk.stem import PorterStemmer
Stemmer = PorterStemmer()
data['stemmed'] = data['Review'].apply(lambda x: " ".join([Stemmer.stem(word) for word in word_tokenize(x)]))
print(data['stemmed'])
data['stemmed'].value_counts()
#Lemmatization
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
data['Lemmatized'] = data['Review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in word_tokenize(x)]))
print(data['Lemmatized'])
#Remove numbers
import re
data['No_Numbers'] = data['Review'].apply(lambda x: re.sub(r'\d+',' ',x))
print(data['No_Numbers'])
data['cleaned_text'] = data['Review'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]',' ',x))
print(data['cleaned_text'])
data.info()
#Normalization
!pip install contractions
import contractions
data['Expanded'] = data['Review'].apply(contractions.fix)
print(data['Expanded'])
!pip install emoji
import emoji
data['Emoji'] = data['Review'].apply(emoji.demojize)
print(data['Emoji'])
#Removinh HTML tags
!pip install beautifulsoup4
from bs4 import BeautifulSoup
data['Cleaned'] = data['Review'].apply(lambda x: BeautifulSoup(x,"html.parser").get_text())
print(data['Cleaned'])