import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample

train_df = pd.read_csv('data/mitbih_train.csv', header=None)
test_df = pd.read_csv('data/mitbih_test.csv', header=None)


def get_padding_start_index(row):
    start_index = 187
    for i in row[::-1]:
        if i != 0.0:
            return start_index

        start_index -= 1


def plot_scan(df, i):
    plt.figure()
    plt.plot(df.iloc[i, :187])


def stretch(signal):
    stretched = [0.0 for i in range(187)]
    for i in range(187):
        scaled_i = i * 2
        if scaled_i < len(stretched):
            stretched[scaled_i] = signal.iloc[i]
        else:
            break

    for i in range(187):
        if not stretched[i]:
            count = 2
            for j in range(i + 1, 187):
                if stretched[j]:
                    stretched[i] = (stretched[i - 1] + stretched[j]) / count
                    break
                count += 1

    return stretched


def squeeze(signal):
    squeezed = []
    i = 0
    while i < len(signal):
        squeezed.append(signal[i])
        i += 2

    for i in range(187 - len(squeezed)):
        squeezed.append(0.0)

    return squeezed


flipped_list = []
noise_list = []
shuffled_list = []
stretched_list = []
squeezed_list = []

segment_length = 8  # Length of segments to get shuffled

# Flip, add noise, shuffle, squeeze and stretch
print("Start Flip, add noise, shuffle, squeeze and stretch for training data")
for row_i in range(len(train_df)):
    padding_i = get_padding_start_index(train_df.iloc[row_i, :187])

    flipped_list.append(train_df.iloc[row_i, 0:padding_i].iloc[::-1].tolist() +
                        train_df.iloc[row_i, padding_i:].tolist())

    noise_list.append(train_df.iloc[row_i, 0:padding_i].transform(lambda x: x + np.random.normal(0, 0.1)).tolist() +
                      train_df.iloc[row_i, padding_i:].tolist())  # Change 0.1 for more or less noise

    n_segments = int(padding_i / segment_length)
    shuffled_index = sample(range(0, n_segments), n_segments)
    shuffled_row_list = []
    for shuffled_i in shuffled_index:
        start_i = shuffled_i * segment_length
        if shuffled_i == n_segments - 1:
            end_i = (shuffled_i * segment_length) + segment_length + (padding_i - (n_segments * segment_length))
        else:
            end_i = (shuffled_i * segment_length) + segment_length
        shuffled_row_list += train_df.iloc[row_i, start_i:end_i].tolist()

    shuffled_list.append(shuffled_row_list + train_df.iloc[row_i, padding_i:].tolist())

    stretched_list.append(stretch(train_df.iloc[row_i, :187]) + [train_df.iloc[row_i, 187]])
    squeezed_list.append(squeeze(train_df.iloc[row_i, :187]) + [train_df.iloc[row_i, 187]])

train_df_flipped = pd.DataFrame(flipped_list)
train_df_noise = pd.DataFrame(noise_list)
train_df_shuffled = pd.DataFrame(shuffled_list)
train_df_stretched = pd.DataFrame(stretched_list)
train_df_squeezed = pd.DataFrame(squeezed_list)

flipped_list = []
noise_list = []
shuffled_list = []
stretched_list = []
squeezed_list = []

segment_length = 8
print("Start Flip, add noise, shuffle, squeeze and stretch for testing data")
for row_i in range(len(test_df)):
    padding_i = get_padding_start_index(test_df.iloc[row_i, :187])

    flipped_list.append(test_df.iloc[row_i, 0:padding_i].iloc[::-1].tolist() +
                        test_df.iloc[row_i, padding_i:].tolist())

    noise_list.append(test_df.iloc[row_i, 0:padding_i].transform(lambda x: x + np.random.normal(0, 0.1)).tolist() +
                      test_df.iloc[row_i, padding_i:].tolist())  # Change 0.1 for more or less noise

    n_segments = int(padding_i / segment_length)
    shuffled_index = sample(range(0, n_segments), n_segments)
    shuffled_row_list = []
    for shuffled_i in shuffled_index:
        start_i = shuffled_i * segment_length
        if shuffled_i == n_segments - 1:
            end_i = (shuffled_i * segment_length) + segment_length + (padding_i - (n_segments * segment_length))
        else:
            end_i = (shuffled_i * segment_length) + segment_length
        shuffled_row_list += test_df.iloc[row_i, start_i:end_i].tolist()

    shuffled_list.append(shuffled_row_list + test_df.iloc[row_i, padding_i:187].tolist())

    stretched_list.append(stretch(test_df.iloc[row_i, :187]) + [test_df.iloc[row_i, 187]])
    squeezed_list.append(squeeze(test_df.iloc[row_i, :187]) + [test_df.iloc[row_i, 187]])

test_df_flipped = pd.DataFrame(flipped_list)
test_df_noise = pd.DataFrame(noise_list)
test_df_shuffled = pd.DataFrame(shuffled_list)
test_df_stretched = pd.DataFrame(stretched_list)
test_df_squeezed = pd.DataFrame(squeezed_list)

# Negate
print("Negating training data")
train_df_negated = train_df.iloc[:, :187].transform(lambda x: x * -1)
train_df_negated.loc[:, 187] = train_df.iloc[:, 187].to_numpy()

print("Negating testing data")
test_df_negated = test_df.iloc[:, :187].transform(lambda x: x * -1)
test_df_negated.loc[:, 187] = test_df.iloc[:, 187].to_numpy()

# Scale
print("Scaling training data")
scale_factor = 0.1  # Change for more or less scaling
train_df_scaled = train_df.iloc[:, :187].transform(lambda x: x * scale_factor)
train_df_scaled.loc[:, 187] = train_df.iloc[:, 187].to_numpy()

print("Scaling testing data")
test_df_scaled = test_df.iloc[:, :187].transform(lambda x: x * scale_factor)
test_df_scaled.loc[:, 187] = test_df.iloc[:, 187].to_numpy()

"""
# Save transformed data
train_df_flipped.to_csv('data/transformations/flipped_train.csv', header=False, index=False)
train_df_noise.to_csv('data/transformations/noise_train.csv', header=False, index=False)
train_df_shuffled.to_csv('data/transformations/shuffled_train.csv', header=False, index=False)
train_df_stretched.to_csv('data/transformations/stretched_train.csv', header=False, index=False)
train_df_squeezed.to_csv('data/transformations/squeezed_train.csv', header=False, index=False)
train_df_negated.to_csv('data/transformations/negated_train.csv', header=False, index=False)
train_df_scaled.to_csv('data/transformations/scaled_train.csv', header=False, index=False)

test_df_flipped.to_csv('data/transformations/flipped_test.csv', header=False, index=False)
test_df_noise.to_csv('data/transformations/noise_test.csv', header=False, index=False)
test_df_shuffled.to_csv('data/transformations/shuffled_test.csv', header=False, index=False)
test_df_stretched.to_csv('data/transformations/stretched_test.csv', header=False, index=False)
test_df_squeezed.to_csv('data/transformations/squeezed_test.csv', header=False, index=False)
test_df_negated.to_csv('data/transformations/negated_test.csv', header=False, index=False)
test_df_scaled.to_csv('data/transformations/scaled_test.csv', header=False, index=False)
"""
