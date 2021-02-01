import pandas as pd

class MakeSentence(object):
    """
    Makes sentences of the data of tokens passed to it
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["token"].values.tolist(), s["pos"].values.tolist(), s["class"].values.tolist())]
        self.grouped = self.data.groupby("id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except ValueError:
            return None


if __name__ == '__main__':
    data = pd.read_csv("idiomscorpus.csv", encoding="latin1").fillna(method="ffill")
    get_sent = MakeSentence(data)                                               # instantiate sentence maker
    sentences = [" ".join(s[0] for s in sent) for sent in get_sent.sentences]   # concat data (originally in tokens) into sentences
    labels = [[s[2] for s in sent] for sent in get_sent.sentences]      # construct true labels for each sentence
    tags_vals = list(set(data["class"].values))                           # generate set of unique labels
    vocab = list(set(data["token"].values))                              # generate vocab/unique data vales
    tag_to_ix = {t: i for i, t in enumerate(tags_vals)}                 # dictionary of labels/tags
    word_to_ix = {j: k for k, j in enumerate(vocab)}                    # dictionary of vocab/data
    print(tag_to_ix)

