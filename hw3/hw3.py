import os
import numpy as np
import pickle
import random


def predict(w, test_data):
    pred = []
    for dat in test_data:
        x = convert_to_sbow(dat[0])
        s = score(w, x)
        if s > 0:
            pred.append(1)
        elif s < 0:
            pred.append(-1)
        else:
            if np.random.rand() < 0.5:
                pred.append(-1)
            else:
                pred.append(1)
    return pred


def score(w, x):
    return dot_product(w, x)


def hinge_loss(x, y, w):
    """
    Compute max(0, 1- y w^Tx)
    @param list x: feature vector
    @param list y: corresponding ground truth label
    @param dict w: sparse representaiton of parameter vector
    @return float
    """
    return max([0, 1 - y * dot_product(w, x)])


def cost(training_data, w, lmbd):
    J = 0.0
    for dat in training_data:
        x = convert_to_sbow(dat[0])  # load review as sBoW
        y = dat[1]  # load label positive/negative

        J += np.maximum(0.0, 1.0 - y * dot_product(w, x))
    J += (lmbd / 2) * dot_product(w, w)
    return J


def train_default(training_data, lmbd=1):
    w = {}
    J_pre = 100.0
    J = 0.0
    t = 0
    epochs = 0
    while abs(J - J_pre) > 1:
        random.shuffle(training_data)
        epochs += 1
        for dat in training_data:
            x = convert_to_sbow(dat[0])  # load review as sBoW
            y = dat[1]  # load label positive/negative

            t = t + 1
            eta_t = 1 / (lmbd * t)
            margin = y * dot_product(w, x)
            if margin < 1:
                scalar_multiply(1 - eta_t * lmbd, w)
                increment(w, eta_t * y, x)
            else:
                scalar_multiply(1 - eta_t * lmbd, w)
        J_pre = J
        J = cost(training_data, w, lmbd)
        print(f"Epoch: {epochs}, Cost: {J}")
    return w


def train_optimized(training_data, lmbd=0, epochs=5):
    W = {}
    J_pre = 100.0
    J = 0.0
    t = 0
    s = 1
    epochs = 0
    while abs(J - J_pre) > 1:
        random.shuffle(training_data)
        epochs += 1
        for dat in training_data:
            x = convert_to_sbow(dat[0])  # load review as sBoW
            y = dat[1]  # load label positive/negative
            t += 1
            eta_t = 1 / (lmbd * t)
            s *= 1 - eta_t * lmbd
            if s == 0:
                s = 1
                W = dict()

            margin = y * s * dot_product(W, x)

            if margin < 1:
                increment(W, (1 / s) * eta_t * y, x)
        w = dict()
        for f, v in W.items():
            w[f] = v * s
        J_pre = J
        J = cost(training_data, w, lmbd)
        print(f"Epoch: {epochs}, Cost: {J}")

    return w


def scalar_multiply(scale, d):
    """
    @param float scale: a scalar to scale the sparse vector with.
    @param dict d: sparse vector in the form of a dict
    @return float: scaled d
    """
    for f, v in d.items():
        d[f] = scale * v


def dot_product(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if not d1 or not d2:
        return 0
    if len(d1) < len(d2):
        return dot_product(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def convert_to_sbow(data: list) -> dict:
    sbow = {}
    for item in data:
        sbow[item] = sbow.get(item, 0) + 1

    return sbow


def folder_list(path, label):
    """
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    """
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path, infile)
        r = [read_data(file)]
        r.append(label)
        review.append(r)
    return review


def read_data(file):
    """
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    """
    f = open(file)
    lines = f.read().split(" ")
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(
        lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines
    )
    words = filter(None, words)
    return list(words)


###############################################
######## YOUR CODE STARTS FROM HERE. ##########
###############################################


def shuffle_data():
    """
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    """
    pos_path = "hw3/data/pos"
    neg_path = "hw3/data/neg"

    pos_review = folder_list(pos_path, 1)
    neg_review = folder_list(neg_path, -1)

    review = pos_review + neg_review
    random.shuffle(review)

    pickle.dump(review, open("hw3/data/reviews.p", "wb"))


"""
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure.
*Check it out. https://wiki.python.org/moin/UsingPickle
"""


def main():
    # shuffle_data() # loads raw review data, shuffles it and svaes to pickle
    reviews = pickle.load(open("hw3/data/reviews.p", "rb"))  # load shuffled reviews
    training_data = reviews[:1500]
    validation_data = reviews[1500:]

    # w = train_default(training_data, lmbd=1)
    # pred = predict(w, validation_data)
    w = train_optimized(training_data, lmbd=0.5)
    pred = predict(w, validation_data)

    correct = 0
    incorrect = 0
    for idx in range(len(pred)):
        if pred[idx] == validation_data[idx][1]:
            correct += 1
        else:
            incorrect += 1

    print(correct)
    print(incorrect)


if __name__ == "__main__":
    main()
