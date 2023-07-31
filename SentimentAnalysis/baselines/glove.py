import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

DIM = 100


# Load glove embeddings
f_embeddings = open("./twitter-datasets/glove.6B.100d.txt")
embeddings = {}
for line in f_embeddings:
    values = line.split()
    word = values[0]
    vector = np.array(values[1:])
    embeddings[word] = vector

"""
    Load tweet dataset and convert it to embeddings

    Returns: array of embeddings and the original text array
"""
def load_embeddings(filename, max=300000):
    f = open(filename)
    pos = []
    original = []
    lengths = []
    tokens = []
    j = 0
    for line in f:
        num_tokens = 0
        if j > max:
            break
        values = line[:len(line)-1].split(' ')
        for val in values:
            if val == '<user>' or val == '<url>':
                num_tokens += 1
        original.append(line)
        values = [val for val in values if val != '<user>' and val in embeddings.keys()]
        for i in range(len(values)):
            if values[i][0] == '#':
                values[i] = values[i][1:]
        values = [val for val in values if val in embeddings.keys()]
        ls = np.zeros((len(values), DIM))
        for i in range(len(values)):
            ls[i] = embeddings[values[i]]
        #print(ls.shape)
        mean = np.mean(ls, axis=0)
        #print(mean.shape)
        pos.append(mean)
        lengths.append(len(values))
        tokens.append(num_tokens)
        j += 1
    return np.array(pos), original, lengths, tokens

# load tweets and convert them to embeddings
pos, original_pos, pos_lengths, tokens_pos = load_embeddings("./twitter-datasets/train_pos_full.txt")
neg, original_neg, neg_lengths, tokens_neg = load_embeddings("./twitter-datasets/train_neg_full.txt")

vectors = np.concatenate((pos, neg), axis=0)
original = np.concatenate((original_pos, original_neg), axis=0)[~np.isnan(vectors).any(axis=1)]
labels = np.concatenate((np.ones((pos.shape[0], 1)), np.zeros((neg.shape[0], 1))), axis=0)
lengths = np.concatenate((pos_lengths, neg_lengths), axis=0)[~np.isnan(vectors).any(axis=1)]
tokens = np.concatenate((tokens_pos, tokens_neg), axis=0)[~np.isnan(vectors).any(axis=1)]
labels = labels[~np.isnan(vectors).any(axis=1)]
vectors = vectors[~np.isnan(vectors).any(axis=1)]

# checkpoint results / comment loading from checkpoint
print(vectors.shape, original.shape, labels.shape, lengths.shape, tokens.shape)
np.save("vectors", vectors)
np.save("labels", labels)
np.save("original", original)
np.save("lengths", lengths)
np.save("tokens", tokens)

vectors = np.load("vectors.npy")
labels = np.load("labels.npy")
original = np.load("original.npy")
lengths = np.load("lengths.npy")
tokens = np.load("tokens.npy")

# split into train/val dataset
indices = np.arange(len(labels))
X_train, X_test, y_train, y_test, _, test_indices = train_test_split(vectors, labels, indices, test_size=0.4, shuffle=True, random_state=42)

np.save("test_indices", test_indices)


# fit Logistic Regression classifier
rf_classifier = LogisticRegression(random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# checkpoint model
joblib.dump(rf_classifier, "forest.joblib")
rf_classifier = joblib.load("forest.joblib")

y_pred = rf_classifier.predict(X_test)

# compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

y_test = y_test.flatten()
y_pred = y_pred.flatten()

# Get the grap of accuracy per tweet length
lengths_test = lengths[test_indices]
def get_accuracy_bin_length():
    bins_f = np.zeros((50,))
    bins_t = np.zeros((50,))
    for i in range(len(lengths_test)):
        l = lengths_test[i] // 5
        if y_test[i] != y_pred[i]:
            bins_f[l] += 1
        else:
            bins_t[l] += 1
    bins_accuracy = np.zeros((50,))
    for i in range(len(bins_accuracy)):
        bins_accuracy[i] = bins_t[i] / (bins_f[i] + bins_t[i])
    return bins_accuracy

acc = get_accuracy_bin_length()[:9]

plt.ylabel("Accuracy")
plt.xlabel("Tweet length (in number of words)")
plt.plot(np.arange(9) * 5, acc)

special_tokens_test = tokens[test_indices]
print(special_tokens_test[:10])

# get the graph of accuracy per number of <user> and <url> in a tweet
# this number is then normalized by tweet length
def get_accuracy_bin_tokens():
    bins_f = np.zeros((100,))
    bins_t = np.zeros((100,))
    for i in range(len(special_tokens_test)):
        l = special_tokens_test[i] // lengths_test[i]
        if y_test[i] != y_pred[i]:
            bins_f[l] += 1
        else:
            bins_t[l] += 1
    bins_accuracy = np.zeros((100,))
    for i in range(len(bins_accuracy)):
        bins_accuracy[i] = bins_t[i] / (bins_f[i] + bins_t[i])
    return bins_accuracy

tok_acc = get_accuracy_bin_tokens()

plt.plot(tok_acc)