import pymorphy2
import pandas as pd

reviews_data = pd.read_csv("reviews/reviews.сsv", delimiter='|', header=None, encoding='utf-8', quoting=1)

reviews_data.columns = ['rating', 'review']

morph = pymorphy2.MorphAnalyzer()

def text_preprocessing(text):
    print(text)
    final_vect = []
    try:
        text = text.lower()
        alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' #удаляем все, кроме русских символов
        cleaned_text = ''
        for char in text:
            if (char.isalpha and char[0] in alphabet) or (char == ' '):
                cleaned_text += char
        for word in cleaned_text.split():
            final_vect.append(morph.parse(word)[0].normal_form)
    except:
        pass
    return ' '.join(final_vect)

reviews_data['review'] = reviews_data['review'].apply(text_preprocessing)

reviews_data.to_csv('reviews/processed_reviews.csv')