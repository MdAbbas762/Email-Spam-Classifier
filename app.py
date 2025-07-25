import pickle
from nltk.corpus import stopwords
import nltk
# nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

input_email = input("Enter the email ")

    # 1. preprocess
transformed_email = transform_text(input_email)
    # 2. vectorize
vector_input = tfidf.transform([transformed_email])
    # 3. predict
result = model.predict(vector_input)[0]
    # 4. Display
if result == 1:
    print("Spam")
else:
    print("Not Spam")