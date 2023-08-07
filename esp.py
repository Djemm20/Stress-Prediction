import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('data.csv')

data['text_sub'] = data['text'] + ' ' + data['subreddit']

X = data['text_sub']
y = data['label']

lemmatizer = WordNetLemmatizer()


def preprocess_dataset(dataset):
    preprocessed_data = []
    for sentence in dataset:
        sentence = re.sub(r'#(\w+)', r'\1', sentence)
        sentence = re.sub(r'@(\w+)', r'\1', sentence)
        sentence = re.sub(r'[?!()]', '', sentence)
        tokens = word_tokenize(sentence)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        preprocessed_text = ' '.join(lemmatized_tokens)
        preprocessed_data.append(preprocessed_text)
    return preprocessed_data


X = preprocess_dataset(X)

# split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# learn data conversion to binary
cv = TfidfVectorizer(binary=True)
cv.fit_transform(X_train)

# convert features to binary
X_train = cv.transform(X_train)
X_test = cv.transform(X_test)

# instance of logistic regression model
log_reg = LogisticRegression()

# train data
model = log_reg.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
