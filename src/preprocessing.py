
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# installation des ressources si ça n'a jamais été fais sur la machine 
# nltk.download()


stop_words = set(stopwords.words('english'))  
lemmatizer = WordNetLemmatizer() 

def clean_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_text(text):
    
    # 1. Supprimer les URLs, mentions, hashtags, emojis
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = re.sub(r"#\w+", "", text)                    # hashtags
    text = re.sub(r"[^\x00-\x7F]+", "", text)           # emojis et caractères non-ASCII

    # 2. Mettre en minuscules
    text = text.lower()

    # 5. Supprimer les stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Retourner le texte nettoyé
    return " ".join(tokens)

def tokenize(text):
    """Tokenise le texte en une liste de mots."""
    return word_tokenize(text)

def lemmatize(tokens):
    """Lemmatisation d'une liste de tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]


