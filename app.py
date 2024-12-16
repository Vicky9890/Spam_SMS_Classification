import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


nltk.download('stopwords')
nltk.download('wordnet')

cleandata = pickle.load(open("clean_data.pkl", "rb"))
tweetdata = pickle.load(open("dataset.pkl", "rb"))

st.header("Spam  SMS Detection Application")

select = st.text_input("Enter SMS:")

wnl=WordNetLemmatizer()


def cleaning(text):
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text) #filtering special characters and numbers
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemm_words = [wnl.lemmatize(word) for word in filtered_words]
    text = ' '.join(lemm_words)
    
    return text

tfid = TfidfVectorizer(max_features = 500)
mnb = MultinomialNB()

X = tfid.fit_transform(cleandata)

mnb.fit(X, tweetdata)



if st.button("Submit"):
    clean_text = cleaning(select)
    transformed_text = tfid.transform([clean_text]).toarray()
    pred = mnb.predict(transformed_text)
    
    
    result =[]
    if pred == 1:
        result = "This is a spam message"
        st.image("https://img.freepik.com/free-vector/stylish-fraud-warning-background-protect-yourself-from-phishing-scams_1017-43350.jpg?w=826&t=st=1729321356~exp=1729321956~hmac=5c0db0f33d030373e48d1a7ecd322422864b0b50c653c5b0f4d94a2126482a98 ")
    else:
        result = "This is a Ham (Normal) message"
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxfozCISn_IlXc1v_r8CKDg5GtwpdYiJLYBg&s")
               
    st.write("Prediction:", result)    
        
