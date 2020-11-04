from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras import regularizers
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
import pandas as pd
from keras.models import model_from_json
from matplotlib import pyplot

#считываем обработанный датасет
dataset = pd.read_csv('reviews/processed_reviews.csv', index_col=0).dropna()
dataset.reset_index(drop=True,inplace=True)

X = dataset['review']
y = dataset['rating']
x_train, x_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=2222)
x_validation, x_test, y_validation, y_test = train_test_split(x_other, y_other, test_size=0.95, random_state=2222)

print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),
                                                                             (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
                                                                            (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))

#размечаем слова по н-грамам, считаем tf-idf для каждого слова
tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,3), max_features=800000)
tfidf_vectorizer.fit(X)
x_train_tfidf = tfidf_vectorizer.transform(x_train)
x_validation_tfidf = tfidf_vectorizer.transform(x_validation).toarray()

seed = 7
np.random.seed(seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].toarray()
        y_batch = y_data[y_data.index[index_batch]]
        counter += 1
        yield X_batch,y_batch
        if (counter > number_of_batches):
            counter=0


#настройки модели нейронной сети
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=800000))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(generator=batch_generator(x_train_tfidf, y_train, 128),
                    epochs=5, validation_data=(x_validation_tfidf, y_validation),
                    steps_per_epoch=x_train_tfidf.shape[0]/128)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#сохраняем весы
model.save_weights("model.h5")


#ниже код в случае, когда уже есть сохраненные весы

'''

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#считаем метрики

x_tftest = tfidf_vectorizer.transform(x_test)
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


from sklearn.metrics import accuracy_score,roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



# predict probabilities for test set
yhat_probs = loaded_model.predict(x_tftest, verbose=0)
# predict crisp classes for test set
yhat_classes = loaded_model.predict_classes(x_tftest, verbose=0)


yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)


# ROC AUC
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)



# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, yhat_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
'''


#Проверка работы обученной модели, проверяем на положительных и отрицательных отзывах с нескольких приложений

x_lol = ['Благодарю всех, кто делится знаниями.',
        'Спасибо всем, кто делится знаниями! Хоть всё и по-английски, но ведь есть и курсы по изучению этого самого английского! Буду пользоваться!',
        'На новом айпад мини 2019приложение вылетает через10 сек после открытия.']
        
x_tappstore = ['Надо упрощать, а не усложнять!!Вы сделали приложение настолько тяжелым и неповоротливым, что на таких старых устройствах как iPad 2 программа Numbers тормозит и не хочет обновляться. А при удалении программы, потворно ее уже не поставишь, так как требуется iOS 10, которая заведомо на iPad 2 не ставится.Для пользователей, которые в бытность iPad 2 покупали эту программу за бешеные деньги в AppStore, это хамство и жульничество.В итоге на iPad 2 с iOS 9 не поставить Numbers, и купленный планшет можно просто выбрасывать за бесполезностью.',
        'Ошибка После обновления не могу открыть сохранённые документы, приложение просит восстановить предыдущую версию файла, что делать?!',
        'Отлично все Работает!!!Открывает много чего другие не могут!!!',
'Класс! Благодарю всех, кто делится знаниями.',
'Замечательное приложение!Спасибо всем, кто делится знаниями! Хоть всё и по-английски, но ведь есть и курсы по изучению этого самого английского! Буду пользоваться!',
               'Бесполезная программа.'
               ]

pos_rate = 0.65

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
x_tappstore = tfidf_vectorizer.transform(x_tappstore)

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(x_s, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

prediction = loaded_model.predict(x_tappstore)
print(prediction)
i = 1
for predict in prediction:
    if predict>pos_rate:
        print('#',i, ' - Положительный')
    else:
        print('#', i, ' - Отрицательный')
    i+=1