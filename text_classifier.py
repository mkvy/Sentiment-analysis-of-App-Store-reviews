from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#считываем обработанный датасет
dataset = pd.read_csv('reviews/processed_reviews.csv', index_col=0).dropna()

tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,3))

X = tfidf_vectorizer.fit_transform(dataset['review'])
y = dataset['rating']



#разбиваем выборку
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

#наивный байес
clf = MultinomialNB()
NB_result = cross_val_score(clf, X, y, cv=cv, n_jobs=-1).mean()
clf.fit(X, y)
#print('Naive Bayes:', NB_result)

import pymorphy2

morph = pymorphy2.MorphAnalyzer()

def text_preprocessing(string):
   # print(string)
    final_vect = []
    try:
        text = string.lower()
        alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' #deleting emojis, links etc
        cleaned_text = ''
        for char in text:
            if (char.isalpha and char[0] in alphabet) or (char == ' '):
                cleaned_text += char
        for word in cleaned_text.split():
            final_vect.append(morph.parse(word)[0].normal_form)
    except:
        pass
    return ' '.join(final_vect)



def predict(strings):
    #print(text_preprocessing(string))
    for string in strings:
        tf_transformer = TfidfTransformer(use_idf=True).fit(X)
        new_doc = [text_preprocessing(string)]
        X_new_counts = tfidf_vectorizer.transform(new_doc)
        X_new_tfidf = tf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)
        print(predicted)

#выбрали отзывы для проверки
revs=['Благодарю всех, кто делится знаниями.',
        'Спасибо всем, кто делится знаниями! Хоть всё и по-английски, но ведь есть и курсы по изучению этого самого английского! Буду пользоваться!',
        'На новом айпад мини 2019приложение вылетает через10 сек после открытия.']
#выводим ответ
predict(revs)

