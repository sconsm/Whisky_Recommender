import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import warnings
import string
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objs as go
warnings.filterwarnings('ignore')
import plotly.figure_factory as ff
from mlxtend.plotting import plot_confusion_matrix 
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.preprocessing import text, sequence 
from tensorflow.keras.models import Sequential
import requests 
from sklearn.decomposition import NMF
from tqdm import tqdm

# Function to convert  
def listToString(s): 
    # initialize an empty string
    str1 = " " 
    # return string  
    return (str1.join(s))


#Initialize
n_samples = 2000
n_features = 1000
n_components = 5
n_top_words = 10

####st.title('Scotch Whisky Recommender:')
st.image('plastic-charity-banner-mobile.jpeg',width=750)
st.markdown("<h1 style='text-align: center; color: maroon;'>Scotch Whisky Recommender</h1>", unsafe_allow_html=True)
st.sidebar.title('Scotch Whisky Recommender:')
#st.markdown("<h1 style='text-align: center; color: maroon;'>Scotch Whisky Recommender</h1>", 

st.sidebar.image('images.jpeg',width=300)
st.markdown("<h1 style='; color: darkgoldenrod;'>A 'Neat' Approach to Find Your Ideal Scotch   Using NLP</h1>", unsafe_allow_html=True)
st.sidebar.title('A NLP Approach to Creating a Recommendation System that Produces Your Ideal Scotch')
#st.markdown("<h2 style='text-align: center; color: maroon;'>A NLP Approach to Creating a Recommendation System that Produces Your Ideal Scotch</h1>", unsafe_allow_html=True)
st.subheader('This Application Displays EDA and NMF Text Recommendations for Scotch Whiskies From A *Whisky Advocate* Kaggle Dataset')
st.sidebar.text('This Application Displays EDA and NMF Text Recommendations for Scotch Whisky')
st.sidebar.title('Visualization Selector')
#st.markdown("<h2 style='text-align: center; color: maroon;'>Visualization Selector</h1>", unsafe_allow_html=True)


st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
review = pd.read_csv("scotch_review.csv")
review.rename(columns={'Unnamed: 0':'ID'},inplace=True)
#ٌSet Value of Pricest.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
review.at[[19, 95, 410, 1000, 1215], 'price'] = 15000  
review['price'].replace('/liter', '', inplace = True, regex = True) 
review['price'].replace(',', '', inplace = True, regex = True)
review['price'] = review['price'].astype('float')
review['price'] = review['price'].astype('int')
#Set Value of Currency
review['currency'].value_counts()
review.drop('currency', axis = 1, inplace = True)
#Extract Age / Name / Alcohol
review['age'] = review['name'].str.extract(r'(\d+) year')[0].astype(float) 
review['name'] = review['name'].str.replace(' ABV ', '')
review['alcohol%'] = review['name'].str.extract(r"([\(\,\,\'\"\’\”\$] ? ?\d+(\.\d+)?%)")[0]
review['alcohol%'] = review['alcohol%'].str.replace("[^\d\.]", "").astype(float)

#Review Datapip
st.markdown("<h1 style='; color: darkgoldenrod;'>Data Used In Analysis</h1>", unsafe_allow_html=True)
st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)
st.write(review)

#WHISKEY CATEGORY VS THEIR COUNT
dataframe_count=review.groupby(['category']).size().sort_values(ascending=True).reset_index(name="Count")
dataframe_count['Count']=dataframe_count['Count']/sum(dataframe_count['Count'])
fig = go.Figure(
    data=[
    go.Bar(x=dataframe_count.category, y=dataframe_count.Count,marker_color='darkgoldenrod'),
])
fig.update_layout(
    xaxis_title="Category of Whisky",
    yaxis_title="Count per Category",
)
fig.update_layout(yaxis_tickformat = '%')
st.markdown("<h1 style='; color: darkgoldenrod;'>CATEGORIES</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='; color: darkgoldenrod;'>Number of Bottles per Category</h1>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)

#Distribution of Review Points
x = review['review.point']
group_labels = 'Distribution of Review Points'
colors=['darkgoldenrod']
fig = ff.create_distplot([x], [group_labels], bin_size=.5, show_rug=False,colors=colors)
fig.update_layout(width=600, height=400,bargap=0.1)
st.markdown("<h1 style='; color: darkgoldenrod;'>Distribution of Review Points</h1>", unsafe_allow_html=True)
st.write("The range of the points given is",review['review.point'].min(),"-",review['review.point'].max())
st.plotly_chart(fig, use_container_width=True)

#DISTRIBUTION OF PRICE
st.markdown("<h1 style='; color: darkgoldenrod;'>Distribution of Price</h1>", unsafe_allow_html=True)
level = st.slider("Choose the maximum price value you wish to display the distribution of.",1,60000)
fig = px.histogram(review, x="price")
fig.update_xaxes(range=[0, level])
fig.update_traces(marker=dict(color="darkgoldenrod"))
st.plotly_chart(fig, use_container_width=True)

#Distribution of Age
fig = px.histogram(review, x="age").update_xaxes(categoryorder="total ascending")
st.markdown("<h1 style='; color: darkgoldenrod;'>Distribution of Age</h1>", unsafe_allow_html=True)
min_value,max_value=st.slider("Choose an age range you wish to display the distribution of.",0,80,value = [0, 70])
fig.update_xaxes(range=[min_value,max_value])
st.markdown('<style>h2{color: blue;}</style>', unsafe_allow_html=True)
fig.update_traces(marker=dict(color="darkgoldenrod"))
st.plotly_chart(fig, use_container_width=True)

#DISTRIBUTION OF ALCOHOL PERCENTAGE
fig = px.histogram(review, x="alcohol%").update_xaxes(categoryorder="total ascending")
st.markdown("<h1 style='; color: darkgoldenrod;'>Distribution of Alcohol Percentage</h1>", unsafe_allow_html=True)
min_value,max_value=st.slider("Choose an alcohol percentage range you wish to display the distribution of.",30,80,value = [40, 70])
fig.update_xaxes(range=[min_value,max_value])
st.markdown('<style>h2{color: blue;}</style>', unsafe_allow_html=True)
fig.update_traces(marker=dict(color="darkgoldenrod"))
st.plotly_chart(fig, use_container_width=True)

#Average Review Point of Different Categories
#1.Box Plot and Median Trend
status = st.radio("Choose how you would like to view the average review points of the different Scotch categories.",("Box Plot","Median Trend"))
if status == 'Box Plot':
    y0 = review[review.category=="Single Malt Scotch"]['review.point'].values
    y1 = review[review.category=="Blended Scotch Whisky"]['review.point'].values
    y2 = review[review.category=="Blended Malt Scotch Whisky"]['review.point'].values
    y3 = review[review.category=="Single Grain Whisky"]['review.point'].values
    y3 = review[review.category=="Single Grain Whisky"]['review.point'].values
    y4 = review[review.category=="Grain Scotch Whisky"]['review.point'].values
    fig = go.Figure()
    fig.add_trace(go.Box(y=y0, name='Single Malt Scotch',marker_color = 'darkgoldenrod'))
    fig.add_trace(go.Box(y=y1, name = 'Blended Scotch Whisky',marker_color = 'darkgoldenrod'))
    fig.add_trace(go.Box(y=y2, name = 'Blended Malt Scotch Whisky',marker_color = 'darkgoldenrod'))
    fig.add_trace(go.Box(y=y3, name = 'Single Grain Whisky',marker_color = 'darkgoldenrod'))
    fig.add_trace(go.Box(y=y4, name = 'Grain Scotch Whisky',marker_color = 'darkgoldenrod'))
    st.markdown("<h1 style='; color: darkgoldenrod;'>Average Review Points per Category</h1>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    y0 = review[review.category=="Single Malt Scotch"]['review.point'].median()
    y1 = review[review.category=="Blended Scotch Whisky"]['review.point'].median()
    y2 = review[review.category=="Blended Malt Scotch Whisky"]['review.point'].median()
    y3 = review[review.category=="Single Grain Whisky"]['review.point'].median()
    y4 = review[review.category=="Grain Scotch Whisky"]['review.point'].median()
    x=["Single Malt Scotch","Blended Scotch Whisky","Blended Malt Scotch Whisky","Single Grain Whisky","Grain Scotch Whisky"]
    y=[y0,y1,y2,y3,y4]
    fig = px.line( x = x ,y = y)
    st.markdown("<h1 style='; color: darkgoldenrod;'>Median Review Points</h1>", unsafe_allow_html=True)
    fig.update_traces(marker=dict(color="darkgoldenrod"))
    st.plotly_chart(fig, use_container_width=True)

##Distribution of Price for Each Category
#y0 = review[review.category=="Single Malt Scotch"]['price'].values
#y1 = review[review.category=="Blended Scotch Whisky"]['price'].values
#y2 = review[review.category=="Blended Malt Scotch Whisky"]['price'].values
#y3 = review[review.category=="Single Grain Whisky"]['price'].values
#y4 = review[review.category=="Grain Scotch Whisky"]['price'].values
#fig = go.Figure()
#fig.add_trace(go.Box(y=y0, name='Single Malt Scotch',marker_color = 'darkgoldenrod'))
#fig.add_trace(go.Box(y=y1, name = 'Blended Scotch Whisky',marker_color = 'darkgoldenrod'))
#fig.add_trace(go.Box(y=y2, name = 'Blended Malt Scotch Whisky',marker_color = 'darkgoldenrod'))
#fig.add_trace(go.Box(y=y3, name = 'Single Grain Whisky',marker_color = 'darkgoldenrod'))
#fig.add_trace(go.Box(y=y4, name = 'Grain Scotch Whisky',marker_color = 'darkgoldenrod'))
#st.markdown("<h1 style='; color: darkgoldenrod;'>Distribution of Price per Category</h1>", #unsafe_allow_html=True)
#st.plotly_chart(fig, use_container_width=True)

#Average to display per Category
st.markdown("<h1 style='; color: darkgoldenrod;'>Category Feature Interactions</h1>", unsafe_allow_html=True)
occupation = st.selectbox("Choose which of the following feature averages you would like to see interact with the category feature.",
["Price",
"Review Point",
"Age",
"Alcohol %"
]
)
if occupation=="Price":
    y1=review[review.category=="Single Malt Scotch"]['price'].mean()
    y2=review[review.category=="Blended Scotch Whisky"]['price'].mean()
    y3=review[review.category=="Blended Malt Scotch Whisky"]['price'].mean()
    y4=review[review.category=="Single Grain Whisky"]['price'].mean()
    y5=review[review.category=="Grain Scotch Whisky"]['price'].mean()
    x=["Single Malt Scotch","Blended Scotch Whisky","Blended Malt Scotch Whisky","Single Grain Whisky","Grain Scotch Whisky"]
    y=[y1,y2,y3,y4,y5]
    fig = go.Figure(
        data=[go.Bar( x=x,y=y,marker_color='darkgoldenrod'),
    ])
    fig.update_layout(
        xaxis_title="Whisky Category",
        yaxis_title="Average Price",
    )
    fig.update_layout(yaxis_tickprefix = '$')
    st.markdown("<h1 style='; color: darkgoldenrod;'>Average Price per Bottle per Category</h1>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
elif  occupation=="Review Point":
    y1=review[review.category=="Single Malt Scotch"]['review.point'].mean()
    y2=review[review.category=="Blended Scotch Whisky"]['review.point'].mean()
    y3=review[review.category=="Blended Malt Scotch Whisky"]['review.point'].mean()
    y4=review[review.category=="Single Grain Whisky"]['review.point'].mean()
    y5=review[review.category=="Grain Scotch Whisky"]['review.point'].mean()
    x=["Single Malt Scotch","Blended Scotch Whisky","Blended Malt Scotch Whisky","Single Grain Whisky","Grain Scotch Whisky"]
    y=[y1,y2,y3,y4,y5]
    fig = go.Figure(
        data=[go.Bar( x=x,y=y,marker_color='darkgoldenrod'),
    ])
    fig.update_layout(
        xaxis_title="Whisky Category",
        yaxis_title="Average Review Points",
    )
    fig.update_layout(yaxis_range=[80,90])
    st.markdown("<h1 style='; color: darkgoldenrod;'>Average Review Point per Bottle per Category</h1>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
elif  occupation=="Age":
    y1=review[review.category=="Single Malt Scotch"]['age'].mean()
    y2=review[review.category=="Blended Scotch Whisky"]['age'].mean()
    y3=review[review.category=="Blended Malt Scotch Whisky"]['age'].mean()
    y4=review[review.category=="Single Grain Whisky"]['age'].mean()
    y5=review[review.category=="Grain Scotch Whisky"]['age'].mean()
    x=["Single Malt Scotch","Blended Scotch Whisky","Blended Malt Scotch Whisky","Single Grain Whisky","Grain Scotch Whisky"]
    y=[y1,y2,y3,y4,y5]
    fig = go.Figure(
        data=[go.Bar( x=x,y=y,marker_color='darkgoldenrod'),
    ])
    fig.update_layout(
        xaxis_title="Whisky Category",
        yaxis_title="Age",
    )
    st.markdown("<h1 style='; color: darkgoldenrod;'>Average Age per Bottle per Category</h1>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
elif  occupation=="Alcohol %":
    y1=review[review.category=="Single Malt Scotch"]['alcohol%'].mean()
    y2=review[review.category=="Blended Scotch Whisky"]['alcohol%'].mean()
    y3=review[review.category=="Blended Malt Scotch Whisky"]['alcohol%'].mean()
    y4=review[review.category=="Single Grain Whisky"]['alcohol%'].mean()
    y5=review[review.category=="Grain Scotch Whisky"]['alcohol%'].mean()
    x=["Single Malt Scotch","Blended Scotch Whisky","Blended Malt Scotch Whisky","Single Grain Whisky","Grain Scotch Whisky"]
    y=[y1,y2,y3,y4,y5]
    fig = go.Figure(
        data=[go.Bar( x=x,y=y,marker_color='darkgoldenrod'),
    ])
    fig.update_layout(
        xaxis_title="Whisky Category",
        yaxis_title="Alcohol %",
    )
    st.markdown("<h1 style='; color: darkgoldenrod;'>Average Alcohol % per Bottle per Category</h1>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

#Word Cloud to display per Category
occupation = st.selectbox("Pick a WordCloud you want to Display. This tells you word frequencies in each category.",["Single Malt Scotch Whisky","Single Grain Whisky",
"Grain Scotch Whisky","Blended Scotch Whisky","Blended Malt Scotch Whisky"
])
if occupation=="Single Malt Scotch Whisky":
    st.markdown("<h1 style='; color: darkgoldenrod;'>Single Malt Scotch Whisky</h1>", unsafe_allow_html=True)
    st.image('Signle_Malt_Scotch.png',width=300)
elif  occupation=="Single Grain Whisky":
    st.markdown("<h1 style='; color: darkgoldenrod;'>Single Grain Whisky</h1>", unsafe_allow_html=True)
    st.image('Signle_Malt_Scotch.png',width=300)
elif  occupation=="Grain Scotch Whisky":
    st.markdown("<h1 style='; color: darkgoldenrod;'>Grain Scotch Whisky</h1>", unsafe_allow_html=True)
    st.image('Grain_Scotch_Whiskey.png',width=300)
elif  occupation=="Blended Scotch Whisky":
    st.markdown("<h1 style='; color: darkgoldenrod;'>Blended Scotch Whisky</h1>", unsafe_allow_html=True)
    st.image('Blended_Scotch_Whiskey.png',width=300)
elif  occupation=="Blended Malt Scotch Whisky":
    st.markdown("<h1 style='; color: darkgoldenrod;'>Blended Malt Scotch Whisky</h1>", unsafe_allow_html=True)
    st.image('Blended_Malt_Scotch_Whiskey.png',width=300)

fig = px.scatter_3d(review, x='review.point', y='age', z='price')
fig.update_traces(marker=dict(color="darkgoldenrod"))
st.markdown("<h1 style='; color: darkgoldenrod;'>3D Feature Correlation Plot</h1>", unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)

#Removing Lower Case
def lowercase_text(text):
    text = text.lower()
    return text
# removing all the unwanted noise (if any)
def remove_noise(text):
    # Dealing with Punctuation
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
#Get_Contractions 
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
#Replace_Contractions
def replace_contractions(text):
    contractions, contractions_re = _get_contractions(contraction_dict)
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)
#Stemming
stemmer = SnowballStemmer("english")
def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)
review['description'] = review['description'].apply(lambda x :lowercase_text(x))
review['description'] = review['description'].apply(lambda x :remove_noise(x))
review['description'] = review['description'].apply(lambda x :replace_contractions(x))
review['description'] = review['description'].apply(lambda x :stemming(x))

def get_recommendation(top, df_all, scores):
    recommendation = pd.DataFrame(columns = ['name', 'category',  'review.point'])
    count = 0
    for i in top:
        recommendation.at[count, 'name'] = df_all['name'][i]
        recommendation.at[count, 'category'] = df_all['category'][i]
        recommendation.at[count, 'review.point'] = df_all['review.point'][i]
        recommendation.at[count, 'price'] = df_all['price'][i]
        count += 1
    return recommendation

def listToString(s):      
    str1 = " "  
    return (str1.join(s)) 

#####################################
#path = st.text_input('Please Enter Description that you want to evaluate')
# MultiSelect
st.markdown("<h1 style='; color: darkgoldenrod;'>Get A Recommendation!</h1>", unsafe_allow_html=True)
location = st.multiselect("Want a delicious Scotch recommendation!? From the list below, choose as many descriptive words (flavors) as you like, and the system will do the rest!",
("soft","toffee","chocolate","sweet","orange","malt","smoke","fruit","caramel",
"spice","apple","lemon","vanilla","orange","dry","ginger","pepper","peat",
"flavors","peach","dark","oak","honey","citrus","aroma","vanilla","wood"))
st.write("You selected the following",len(location),"options:",location)

firstname = st.text_input("You may also choose to enter your own bottle description below.Try to only type things you do like, versus things you don't like.")

stop_words = ['whisky', 'whiskies', 'blend', 'note', 'notes', 'year', 'years', 'old', 'nose', 'finish', 'bottle',
              'bottles', 'bottled', 'along', 'release', 'flavor', 'cask', 'well', 'make', 'mouth', 'palate', 'hint',
              'one', 'bottling', 'distillery', 'quite', 'time', 'date', 'show', 'first'] + list(STOPWORDS)
number_of_recommended_bottles = st.slider('Finally, please tell us how many bottles you want us to recommend for you.')
if st.button("Run"):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_jobid = tfidf_vectorizer.fit_transform((review['description']))
    nmf = NMF(n_components=n_components, random_state=1,alpha=.1, l1_ratio=.5).fit(tfidf_jobid)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    u = 20
    index = np.where(review['ID'] == u)[0][0]
    user_q = review.iloc[[index]]
    input_value=listToString(location)
    input_value=[input_value]
    user_tfidf = tfidf_vectorizer.transform(input_value)
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)
    output2 = list(cos_similarity_tfidf)
    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:number_of_recommended_bottles]
    list_scores = [output2[i][0][0] for i in top]
    m=get_recommendation(top,review, list_scores)
    st.write(m)


def get_recommendation_built_in(top, df_all, scores):
    recommendation = pd.DataFrame(columns = ['name', 'category',  'review.point'])
    count = 0
    for i in top:
        recommendation.at[count, 'name'] = df_all['name'][i+1]
        recommendation.at[count, 'category'] = df_all['category'][i+1]
        recommendation.at[count, 'review.point'] = df_all['review.point'][i]
        recommendation.at[count, 'age'] = df_all['age'][i]
        recommendation.at[count, 'alcohol%'] = df_all['alcohol%'][i]
        count += 1
    return recommendation

st.markdown("<h1 style='; color: darkgoldenrod;'>Built in Data point</h1>", unsafe_allow_html=True)
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_jobid = tfidf_vectorizer.fit_transform((review['description']))
nmf = NMF(n_components=n_components, random_state=1,alpha=.1, l1_ratio=.5).fit(tfidf_jobid)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
u = 20
index = np.where(review['ID'] == u)[0][0]
built_in_data_point = pd.DataFrame(columns = ['name', 'category',  'review.point'])
built_in_data_point.at[0, 'name'] = review['name'].iloc[[index]].item()
built_in_data_point.at[0, 'category'] = review['category'].iloc[[index]].item()
built_in_data_point.at[0, 'review.point'] = review['review.point'].iloc[[index]].item()
built_in_data_point.at[0, 'age'] = review['age'].iloc[[index]].item()
built_in_data_point.at[0, 'alcohol'] = review['alcohol%'].iloc[[index]].item()
st.write(built_in_data_point)
user_q = review['description'].iloc[[index]].item()
user_q = [user_q]
user_tfidf = tfidf_vectorizer.transform(user_q)
cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)
output2 = list(cos_similarity_tfidf)
top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:5]
list_scores = [output2[i][0][0] for i in top]
m=get_recommendation_built_in(top,review, list_scores)
st.markdown("<h1 style='; color: darkgoldenrod;'>Prediction</h1>", unsafe_allow_html=True)
st.write(m)




