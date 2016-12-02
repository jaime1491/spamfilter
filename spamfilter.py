import pandas as pd
import os, sys, getopt, cPickle, csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, train_test_split
from textblob import TextBlob

MESSAGES = pd.read_csv('datas.txt', sep='\t', quoting=csv.QUOTE_NONE, names=["message", "label"])

def tokens(message):
    message = unicode(message, 'utf8')
    return TextBlob(message).words

def lemmas(message):
    message = unicode(str(message), 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

def train_multinomial_nb(messages):
    msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'],
                                                                    test_size=0.9)

    pipeline = Pipeline(
        [('bow', CountVectorizer(analyzer=lemmas)), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])

    params = {
        'tfidf__use_idf': (True, False),
        'bow__analyzer': (lemmas, tokens),
    }
    grid = GridSearchCV(
        pipeline,
        params,
        refit=True,
        n_jobs=-1,
        scoring='accuracy',
        cv=StratifiedKFold(label_train, n_folds=5))

    nb_detector = grid.fit(msg_train, label_train)
    print ""
    predictions = nb_detector.predict(msg_test)
    print ":: Confusion Matrix"
    print ""
    print confusion_matrix(label_test, predictions)
    print "Accuracy:"
    print accuracy_score(label_test, predictions)

    file_name = 'spam_nb_model.pkl'
    with open(file_name, 'wb') as fout:
        cPickle.dump(nb_detector, fout)
    print 'model written to: ' + file_name

def main(argv):
    if (os.path.isfile('spam_nb_model.pkl') == False):
        print ""
        print "Naive Bayes Model....."
        train_multinomial_nb(MESSAGES)

    opts, args = getopt.getopt(argv, "hm:", ["message="])
    print 'spamfilter.py -m <message string>'

    for opt, arg in opts:
        if opt == '-h':
            print 'spam_filter.py -m <message string>'
            sys.exit()
        elif opt in ("-m", "--message"):
            predictioner = predict(arg)
            print 'This message is', predictioner

def predict(message):
    nb_detector = cPickle.load(open('spam_nb_model.pkl'))
    nb_predict = nb_detector.predict([message])[0]
    return nb_predict

if __name__ == "__main__":
    main(sys.argv[1:])