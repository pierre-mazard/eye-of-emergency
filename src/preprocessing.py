import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd 

# installation des ressources si ça n'a jamais été fais sur la machine 
# nltk.download()

# Initialisation
df = pd.read_csv("data//raw//train_tweets.csv", encoding='utf-8')
stop_words = set(stopwords.words('english'))  
lemmatizer = WordNetLemmatizer() 


def preprocess_text(text):
    
    # 1. Supprimer les URLs, mentions, hashtags, emojis
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = re.sub(r"#\w+", "", text)                    # hashtags
    text = re.sub(r"[^\x00-\x7F]+", "", text)           # emojis et caractères non-ASCII

    # 2. Mettre en minuscules
    text = text.lower()

    # 3. Supprimer la ponctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Tokeniser
    tokens = word_tokenize(text)

    # 5. Supprimer les stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Lemmatiser
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Retourner le texte nettoyé
    return " ".join(tokens)

# exemple de test 
test = "This is a tests tweets to sees if the functions works !?? Check out https://example.com @user #hashtag 😊🤣🤣🫡"
print(preprocess_text(test))

processed_csv = df.copy()
processed_csv['text'] = processed_csv['text'].apply(preprocess_text)

processed_csv.to_csv("data//processed//processed_train_tweets.csv", index=False, encoding='utf-8')




