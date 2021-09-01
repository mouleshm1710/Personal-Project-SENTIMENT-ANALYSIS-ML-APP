 
import pickle
import streamlit as st
import nltk
import gensim 
import sklearn 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon') 
import re
from nltk.stem import WordNetLemmatizer
nltk.download('punkt') # dependancy
nltk.download('wordnet') # dependancy   
import numpy as np

pickle_1 = open('glove_vocab.pkl', 'rb')
glove_words = pickle.load(pickle_1)
pickle_1.close()

pickle_2 = open('glove_vectors', 'rb') 
model_key_vectors = pickle.load(pickle_2)
pickle_2.close()

pickle_3 = open('normalizer2.pkl', 'rb') 
minmaxscaler = pickle.load(pickle_3)
pickle_3.close()

pickle_4 = open('log_model2.pkl', 'rb') 
classifier = pickle.load(pickle_4)
pickle_4.close()

minmaxscaler.clip = False

stopwords= ['i','im','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'could','couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'had',\
            "hadn't", 'has', "hasn't", 'have', "haven't", 'isn', "isn't", 'ma', 'might', "mightn't", 'must',\
            "mustn't", 'need', "needn't", 'shan', "shan't",'should',"shouldn't", 'wasn', "wasn't", 'weren', "weren't",\
            "won't",'would', "wouldn't"]

@st.cache() 

# general cleaning of text data 
def preprocessing_txt(text_data):  
    
    preprocessed_text = []         # store the cleaned text
    
    # Remove punctuations & special characters in the tweets
    replaced = re.sub(r'(\<.*\>)','',text_data)
    replaced = re.sub(r'(\(.*\))','',replaced)
    replaced = re.sub('(\s+)',' ',replaced)
    replaced = re.sub('\w+\:+','',replaced)
    replaced = re.sub('[\,\!\/\)\(\-\<\.\?\>\@\*\_\#\;\|\+\=\$\&\:\"]+','',replaced) 

        # Expand the words & remove digits in the tweets
    phrase = re.sub(r"won't", "will not",replaced,flags = re.I)
    phrase = re.sub(r"can\'t", "can not",phrase,flags = re.I)
    phrase = re.sub(r"n\'t", " not", phrase,flags = re.I)
    phrase = re.sub(r"\'re", " are", phrase,flags = re.I)
    phrase = re.sub(r"\'s", " is", phrase,flags = re.I)
    phrase = re.sub(r"\'d", " would", phrase,flags = re.I)
    phrase = re.sub(r"\'ll", " will", phrase,flags = re.I)
    phrase = re.sub(r"\'t", " not", phrase,flags = re.I)
    phrase = re.sub(r"\'ve", " have", phrase,flags = re.I)
    replaced = re.sub(r"\'m", " am", phrase,flags = re.I)
    replaced = re.sub('\d+','',replaced) 

    # Convert to lowercase & strip the trailing spaces 
    replaced = replaced.lower()
    replaced = re.sub('\W+',' ',replaced)
    replaced = ' '.join(e for e in replaced.split() if e not in stopwords)
    replaced = replaced.strip()
    
    # Tokenize: Split the tweet into words
    word_list = nltk.word_tokenize(replaced)
    
    # lemmatize the list of words and join as a sentence
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    
    # finally append the cleaned text to the list
    preprocessed_text.append(lemmatized_output)
    
    return preprocessed_text[0] # return the list

def vectorize(new_tweet):
    
    avg_w2v_vector = []; # the avg-w2v for each sentence is stored in this list
    
    vector = np.zeros(300)          # Dim is 300
    cnt_words =0;                   # num of words with a valid vector in the sentence
    for word in new_tweet.split():   # for each word in a tweet/sentence
        if word in glove_words: 
            vector += model_key_vectors[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vector.append(vector)  # Append the vector

    return avg_w2v_vector[0]



def polarity_scores(new_tweet):
    
    # Obtain the polarity scores of the tweet
    scores = []
    sid = SentimentIntensityAnalyzer() 
    ss = sid.polarity_scores(new_tweet)   # Polarity scores computed
    ss.pop('neu')
    for k in ss.items():
        scores.append(k[1])
        
    return scores


def main():      
# front end elements of the web page 
    st.write("LETS START")
    html_temp = """<div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Tweet Sentiment Prediction ML App</h1> 
    </div>"""
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
    Gender = st.selectbox('Gender',("Male","Female"))
    TwitterID = st.number_input("User's Twitter ID",step = 1)
    Tweet = st.text_input("Enter the tweet") 
     
    if st.button("Predict"): 

       Cleaned_text = preprocessing_txt(Tweet)
       #st.success("We got our cleaned text --> {}".format(Cleaned_text))
       
       Vectorized_output = list(vectorize(Cleaned_text))   # list 
       #st.success("Dimension of vectorized output --> {}".format(len(Vectorized_output)))
       
       Polarity_scores =  polarity_scores(Cleaned_text) # list
       #st.success("Dimension of Polarity scores --> {}".format(len(Polarity_scores)))

       Vectorized_output.extend(Polarity_scores)
       final_vector =  np.array(Vectorized_output) 
       #st.success("Dimension of final_vector --> {}".format(final_vector.shape))
       
       reshaped_array = final_vector.reshape((1,-1))
       
       scaled_vector = minmaxscaler.transform(reshaped_array)

       output = classifier.predict(scaled_vector) 
       prob = classifier.predict_proba(scaled_vector)
       st.success("Hurray :)  we got the result")
       #st.success("The output value --> {}".format(output))
   
       st.success("Probability that the tweet is negative --> {}".format(prob[0][0]))
       st.success("Probability that the tweet is positive --> {}".format(prob[0][1]))
       if output == 'negative':
          st.success("The sentiment of the tweet is NEGATIVE") 

       else:
          st.success("The sentiment of the tweet is POSITIVE") 
        
if __name__ == "__main__":              
    main()                   
