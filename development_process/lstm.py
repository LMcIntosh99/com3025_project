import numpy as np

import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from sklearn.utils import resample

from warnings import filterwarnings
filterwarnings('ignore')


train_df = pd.read_csv('data/mitbih_train.csv', header=None)
test_df = pd.read_csv('data/mitbih_test.csv', header=None)
weight_path = "data/weights_lstm.hdf5"

df_1 = train_df[train_df[187] == 1]
df_2 = train_df[train_df[187] == 2]
df_3 = train_df[train_df[187] == 3]
df_4 = train_df[train_df[187] == 4]
df_0 = (train_df[train_df[187] == 0]).sample(n=20000, random_state=42)

df_1_upsample = resample(df_1, replace=True, n_samples=20000, random_state=123)
df_2_upsample = resample(df_2, replace=True, n_samples=20000, random_state=124)
df_3_upsample = resample(df_3, replace=True, n_samples=20000, random_state=125)
df_4_upsample = resample(df_4, replace=True, n_samples=20000, random_state=126)

train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

y_train = to_categorical(train_df[187], 5)
y_test = to_categorical(test_df[187], 5)

x_train = train_df.iloc[:, :186].values
x_test = test_df.iloc[:, :186].values
x_train = x_train.reshape(len(x_train), x_train.shape[1], 1)
x_test = x_test.reshape(len(x_test), x_test.shape[1], 1)

train_weights = False
batch_size = 64
epochs = 20

model = Sequential()
model.add(LSTM(64, input_shape=(186, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

if train_weights:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint])

else:
    model.load_weights(weight_path)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print(score, acc)

y_ptb_normal = to_categorical(ptb_normal[187], 5)
y_ptb_abnormal = to_categorical(ptb_abnormal[187], 5)

x_ptb_normal = ptb_normal.iloc[:, :186].values
x_ptb_abnormal = ptb_abnormal.iloc[:, :186].values
x_ptb_normal = x_ptb_normal.reshape(len(x_ptb_normal), x_ptb_normal.shape[1])

print("normal")
score, acc = model.evaluate(x_ptb_normal, y_ptb_normal, batch_size=batch_size)
n_correct = acc * len(x_ptb_normal)


correct = 0
for scan_i in range(len(x_ptb_abnormal)):
    ab_test = pd.DataFrame(x_ptb_abnormal).values[scan_i]
    ab_test = ab_test.reshape(1, x_ptb_abnormal.shape[1])
    prediction = model.predict(ab_test)

    predictions = model.predict([ab_test])[0]
   # print(predictions)
    if sum(predictions[1:]) > predictions[0]:
        correct += 1

accuracy = correct / len(x_ptb_abnormal)
print("Total Accuracy = {}".format((correct + n_correct)/(len(x_ptb_abnormal) + len(x_ptb_normal))))
