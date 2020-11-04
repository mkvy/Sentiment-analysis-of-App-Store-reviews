# Sentiment-analysis-of-App-Store-reviews
EN

Sentiment analysis using machine learning alghoritms to predict positive/negative reviews.
File text_classifier.py contains Naive Bayes model.
File neural_check.py contains realisation of deep feedforward neural net.
The code for collecting the dataset is in get_reviews.py.
The code for preprocessing the dataset is in prepocessing_text.py.


Dataset is located in folder "reviews". Dataset includes about 115000 reviews in Russian from the App Store games category.
Reviews divided into two classes: positive and negative. The ratio of positive to negative reviews is about half.
Accuracy for Naive Bayes model is about 82%.
Accuracy for neural net is about 79%. F1 score is 79%. ROC AUC is 85%. Precision - 79%, Recall - 80.3%.

RU

Анализ тональности с использованием алгоритмов машинного обучения.
В файле text_classifier.py реализована модель наивного Байеса.
В файле neural_check.py реализована модель нейронной сети глубокого обучения.
Код, использованный для сбора датасета находится в файле get_reviews.py.
Код для обработки датасета в файле preprocessing_text.py.

Датасет находится в папке "reviews". Он содержит примерно 115000 отзывов на русском языке, взятых с категории "Игры" в магазине приложений App Store.
Отзывы были размечены на два класса: положительный и отрицательный. Соотношение положительных к отрицательным отзывам составляет примерно половину.
Точность для модели наивного Байеса была равна примерно 82%.
Для нейронных сетей точность составляла примерно 79%. Метрика F1 - 79%, метрика ROC AUC - 85%. Precision - 79%, Recall - 80.3%.
