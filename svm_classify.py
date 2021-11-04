# Tosin Adewumi
"""

"""

import argparse
import torch
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, plot_confusion_matrix, classification_report
import random
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--cuda', default='cuda', action='store_true', help='use CUDA')
parser.add_argument('--ofile1', type=str, default='out_svm_idioms.txt', help='output file')
args = parser.parse_args()

# constants
#batch_size = 3
#epochs = 5
#seed_val = 17
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

class MakeSentence(object):
    """
    Makes sentences of the data of tokens passed to it
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w.lower(), p, t) for w, p, t in zip(s["token"].values.tolist(), s["pos"].values.tolist(), s["class"].values.tolist())]
        self.grouped = self.data.groupby("id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except ValueError:
            return None


def print_plot(index):
    example = df[df.index == index][['sentence', 'labelclass']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Label:', example[1])


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text).text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub('', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    

if __name__ == '__main__':
    device = torch.device(args.cuda)        # initialize device
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            device = torch.device("cuda" if args.cuda else "cpu")
    else:
        device = torch.device("cpu")

    labelclass = []
    #df = pd.read_csv('title_conference.csv')
    df = pd.read_csv("corpus/idiomscorpus.csv", encoding="latin1").fillna(method="ffill")
    
    get_sent = MakeSentence(df)                                               # instantiate sentence maker
    sentences = [" ".join(s[0] for s in sent) for sent in get_sent.sentences]   # concat data (originally in tokens) into sentences
    labels = [[s[2] for s in sent] for sent in get_sent.sentences]      # construct true labels for each sentence
    for numb in range(len(labels)):
        labelclass.append(labels[numb][0])
    corp_dict = {'sentence':sentences, 'labelclass':labelclass}                  # construct dict from the 2 lists
    df = pd.DataFrame(corp_dict)
    df['sentence'] = df['sentence'].apply(clean_text)                           # pre-processing
    #print_plot(30)
    possible_labels = df.labelclass.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)
    df['label'] = df.labelclass.replace(label_dict)                 # replace labels with their nos

    #Data split
    # possible_labels = df.Conference.unique()
    # X = df.Title
    # y = df.Conference

    # label_dict = {}
    # for index, possible_label in enumerate(possible_labels):
    #     label_dict[possible_label] = index
    # label_dict
    # df['label'] = df.Conference.replace(label_dict)

    X = df.sentence
    y = df.labelclass
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 42, stratify=df.label.values)

    # mNB classifier
    nb = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
                ])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy_mnb = accuracy_score(y_pred, y_test)
    class_rp_nmb = classification_report(y_test, y_pred,target_names=set(df.labelclass.tolist()))

    with open(args.ofile1, "a+") as f:
        s = f.write(f'multinomial Naive Bayes:' + "\n")
    print('accuracy %s' % accuracy_mnb)
    print(class_rp_nmb)
    with open(args.ofile1, "a+") as f:
        s = f.write(f'Accuracy: {accuracy_mnb}' + "\n" + f'Classification report \n: {class_rp_nmb}' + "\n")

    # linear SVM
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy_svm = accuracy_score(y_pred, y_test)
    class_rp_svm = classification_report(y_test, y_pred,target_names=set(df.labelclass.tolist()))
    
    with open(args.ofile1, "a+") as f:
        s = f.write(f'Linear SVM:' + "\n")
    print('accuracy %s' % accuracy_svm)
    print(class_rp_svm)
    with open(args.ofile1, "a+") as f:
        s = f.write(f'Accuracy: {accuracy_svm}' + "\n" + f'Classification report \n: {class_rp_svm}' + "\n")

    # # Logistic Regression
    # logreg = Pipeline([('vect', CountVectorizer()),
    #                 ('tfidf', TfidfTransformer()),
    #                 ('clf', LogisticRegression(n_jobs=1, C=1e5)),
    #             ])
    # logreg.fit(X_train, y_train)
    # y_pred = logreg.predict(X_test)
    # accuracy_lgr = accuracy_score(y_pred, y_test)
    # class_rp_lgr = classification_report(y_test, y_pred,target_names=set(df.labelclass.tolist()))

    # with open(args.ofile1, "a+") as f:
    #     s = f.write(f'Logistic Regression:' + "\n")
    # print('accuracy %s' % accuracy_lgr)
    # print(class_rp_lgr)
    # with open(args.ofile1, "a+") as f:
    #     s = f.write(f'Accuracy: {accuracy_lgr}' + "\n" + f'Classification report \n: {class_rp_lgr}' + "\n")

    # # Confusion matrix plot
    # np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #                 ("Normalized confusion matrix", 'true')]
    # for title, normalize in titles_options:
    #     disp = plot_confusion_matrix(logreg, y_test, y_pred,
    #                                 display_labels=set(df.labelclass.tolist()),
    #                                 cmap=plt.cm.Blues,
    #                                 normalize=normalize)
    #     disp.ax_.set_title(title)

    #     print(title)
    #     print(disp.confusion_matrix)

    # plt.show()
