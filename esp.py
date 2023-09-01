# import necessary libraries
import re
import pandas as pd
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# read dataset
data = pd.read_csv('data.csv')

# combine features
data['text_sub'] = data['text'] + ' ' + data['subreddit']

# assign feature and target variable
X = data['text_sub']
y = data['label']

lemmatizer = WordNetLemmatizer()

# preprocess dataset
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
vectorizer = TfidfVectorizer(binary=True)
vectorizer.fit_transform(X_train)

# convert features to binary
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

# instance of logistic regression model
log_reg = LogisticRegression()

# train data
model = log_reg.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)
print(cm)

plt.bar(['accuracy', 'precision', 'recall', 'f1_score'], [accuracy, precision, recall, f1])
plt.xlabel('Metric')
plt.ylabel('Score')
plt.show()
