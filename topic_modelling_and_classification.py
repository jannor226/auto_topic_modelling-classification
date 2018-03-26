import pandas as pd
import numpy as np
import html.parser
import nltk, re, string, collections, pickle, gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, LancasterStemmer, PorterStemmer
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

np.random.seed(10)

wnl = WordNetLemmatizer()
ps = PorterStemmer()
ls = LancasterStemmer()

def read_data():
    '''
    Reads first sheet of the excel spreadsheet and return a number of row
    to be analyzed
    '''
    xl_sheet = pd.read_excel("Exercise_Data Scientist.xlsx", sheet_name=0)
    return xl_sheet.iloc[:500]

def tonkenize_text(tweet):
    tokens = nltk.word_tokenize(tweet)
    return tokens

def remove_special_char(string):
    '''
    Removes numbers, tweet names, links and hastags
    '''
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",string).split())
    return text

def fix_contractions(tweet):
    '''
    fix contractions by converting them into useful forms. This list can be extended
    '''
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    tweet = re.sub(r"\'m", " am", tweet)
    return tweet

def remove_stopwords(tweet):
    '''
    Removes stopwords. Added customs due to topics being extracted
    '''
    stop_words = set(stopwords.words('english'))
    stop_words.update([
    'tesla', 'model', 'car', 'rt', 'r', 'elon', 'musk', 'tsla', 't', 'much',
    'ra', 'could', 'would', 'should', 'still', 'please','say','get','retweet',
    'retwet','therefore','new', 'like','b'
    ])
    tokens = tonkenize_text(tweet)
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens

def replace(old_word):
    '''
    uses regex to find patterns of repeated characters, searches in wordnet
    for a similar word if not available, deletes repetiton
    '''
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_sub = r'\1\2\3'
    if wordnet.synsets(old_word):
        return old_word
    new_word = repeat_pattern.sub(match_sub, old_word)
    return replace(new_word) if new_word != old_word else new_word

def remove_repeated_char(tokens):
    '''
    Removes repeated characters, eg. coverts finalllly to finally
    '''
    corrected_tokens = [replace(word) for word in tokens]
    return corrected_tokens

def pos_to_wordnet_tag(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def pos_tag_text(text):
    tagged_text = nltk.pos_tag(text, tagset='universal')
    tagged_lower_text = [(word.lower(), pos_to_wordnet_tag(pos_tag))
                            for word, pos_tag in tagged_text]
    return tagged_lower_text

def lemmatize(text):
    '''
    Lemmatizing, transforming word to their original form. Uses pos_tag and
    pos_to_wordnet_tag as a comparison
    '''
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
    return lemmatized_tokens

def known(words):
    '''
    searches for popular or matching words from edits
    '''
    return {w for w in words if w in token_count}

def splits(word):
    return [(word[:i], word[i:]) for i in range(len(word)+1)]

def edits0(word):
    '''
    words with zero edits aka correct words
    '''
    return {word}

def edits1(word):
    '''
    words with one edit
    '''
    letter_list = 'abcdefghijklmnopqrstuvwxyz'
    letter_pair = splits(word)
    deletes = [a+b[1:] for (a,b) in letter_pair if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a,b) in letter_pair if len(b) > 1]
    replaces = [a+b+c[1:] for (a,b) in letter_pair for c in letter_list if b]
    inserts = [a+b+c for (a,b) in letter_pair for c in letter_list]

    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    '''
    words with 2 edits
    '''
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def correct(word):
    '''
    compare all edits and find the correct spelling by looking in the corpus
    '''
    candidates = (
                known(edits0(word)) or
                known(edits1(word)) or
                known(edits2(word)) or
                [word]
                )
    return max(candidates, key=token_count.get)

def normalize_corpus(corpus, tokenize=True):
    '''
    Function to normalize all text data by:
    removing special characters, repeating char, numbers, etc
    '''
    # print(corpus)
    corp = corpus.lower()
    corp = remove_special_char(corp)
    corp = fix_contractions(corp)
    corp = re.sub(r'[^\w\s]',' ',corp)
    corp = ''.join([i for i in corp if not i.isdigit()])
    if tokenize == True:
        clean_txt = remove_stopwords(corp)
        norm_txt = remove_repeated_char(clean_txt)
        corrected_txt = [correct(token) for token in norm_txt]
        normalized_corpus = lemmatize(corrected_txt)

    else:
        normalized_corpus = corp

    return " ".join(normalized_corpus)

def evaluate_model_graph(coherences, indices):
    '''
    function to print or plot coherence scores
    '''
    assert len(coherences) == len(indices)
    no = len(coherences)
    x = np.arange(no)
    # plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    # plt.xlabel('Model')
    # plt.ylabel('Coherence Score')
    print(coherences)

dataset = read_data()

raw_corpus = dataset.Contents

#corpus word count for spell check algorithm
token_corp = word_tokenize(str(raw_corpus))
token_count = collections.Counter(token_corp)

norm_corpus = raw_corpus.apply(normalize_corpus)
train_corpus = norm_corpus.apply(remove_stopwords)

#Bigram would been necessary for joining words like new_york so they dont affect the model
# bigram = gensim.models.Phrases(train_corpus)
# train_corpus = [bigram[line] for line in train_corpus]

dictionary = Dictionary(train_corpus)
final_corpus = [dictionary.doc2bow(text) for text in train_corpus]

# Unsupervised learning approach to get the number of topics in this dataset
hdpmodel = HdpModel(corpus=final_corpus, id2word=dictionary)
print(hdpmodel.show_topics())

#Latent Semantic Indeixing, a popular information retreival method which works by decomposing the original matrix of words to maintain key topics
lsimodel = LsiModel(corpus=final_corpus, num_topics=10, id2word=dictionary)
print(lsimodel.show_topics())

#Latent Dirichlet Allocation - famous topic modelling algorithm out there
ldamodel = LdaModel(corpus=final_corpus, num_topics=10, chunksize=100,
                    update_every=1, id2word=dictionary, minimum_probability=0)
print(ldamodel.show_topics())

#Topic Coherence to identify which model is doing better
lsitopics = [[word for word, prob in topic] for topicid, topic in lsimodel.show_topics(formatted=False)]
hdptopics = [[word for word, prob in topic] for topicid, topic in hdpmodel.show_topics(formatted=False)]
ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]

lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=train_corpus, dictionary=dictionary, window_size=10).get_coherence()
hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=train_corpus, dictionary=dictionary, window_size=10).get_coherence()
lda_coherence = CoherenceModel(topics=ldatopics, texts=train_corpus, dictionary=dictionary, window_size=10).get_coherence()

#From our evaluate model we can tell the unsupervised algorithm worked better
evaluate_model_graph([lsi_coherence, hdp_coherence, lda_coherence], ['LSI', 'HDP', 'LDA'])

#assigning topics to clusters using ldamodel
trained_corpus = ldamodel[final_corpus]

#calculating threshold to ensure even distribution i.e 1/no. of clusters
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in trained_corpus]]))

threshold = sum(scores)/len(scores)

#generating cluster of tweets. Using norm corpus bc I envision every tweet to be normalized before prediction
cluster1 = [j for i,j in zip(trained_corpus,norm_corpus) if i[0][1] > threshold]
cluster2 = [j for i,j in zip(trained_corpus,norm_corpus) if i[1][1] > threshold]
cluster3 = [j for i,j in zip(trained_corpus,norm_corpus) if i[2][1] > threshold]
cluster4 = [j for i,j in zip(trained_corpus,norm_corpus) if i[3][1] > threshold]
cluster5 = [j for i,j in zip(trained_corpus,norm_corpus) if i[4][1] > threshold]
cluster6 = [j for i,j in zip(trained_corpus,norm_corpus) if i[5][1] > threshold]
cluster7 = [j for i,j in zip(trained_corpus,norm_corpus) if i[6][1] > threshold]
cluster8 = [j for i,j in zip(trained_corpus,norm_corpus) if i[7][1] > threshold]
cluster9 = [j for i,j in zip(trained_corpus,norm_corpus) if i[8][1] > threshold]
cluster10 = [j for i,j in zip(trained_corpus,norm_corpus) if i[9][1] > threshold]

# print(cluster1)

'''
For Classification

Step 1: match topics to clusters and create a dataframe
Step 2: remove stopwords and generate features (used tfidf)
Step 3: evaluate model
'''
classification_data = {
                        "topic_1": cluster1,
                        "topic_2": cluster2,
                        "topic_3": cluster3,
                        "topic_4": cluster4,
                        "topic_5": cluster5,
                        "topic_6": cluster6,
                        "topic_7": cluster7,
                        "topic_8": cluster8,
                        "topic_9": cluster9,
                        "topic_10": cluster10,
                    }

class_dataframe = pd.DataFrame.from_dict(classification_data, orient='index').transpose()

#create training dataset for classification
final_class_data = pd.melt(class_dataframe)
final_class_data = final_class_data.dropna()

def prepare_dataset(corpus, labels, test_data_proportion=0.3):
    '''
    creates a train and test split of calssification dataset
    '''
    train_x, test_x, train_y, test_y = tts(
                                            corpus,
                                            labels,
                                            test_size=0.3,
                                            random_state=42)
    return train_x, test_x, train_y, test_y

def tfidf_extractor(corpus, ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(norm='l2',smooth_idf=True,use_idf=True,ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

def get_metrics(true_labels, predicted_labels):
    '''
    prints classifier results
    '''
    print('Accruacy', np.round(metrics.accuracy_score(true_labels,predicted_labels),2))
    print('Precision', np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted'),2))
    print('F1 Score', np.round(metrics.f1_score(true_labels,predicted_labels,average='weighted'),2))
    print('Recall', np.round(metrics.recall_score(true_labels,predicted_labels,average='weighted'),2))

def train_predict_evaluate(classifier, train_features, train_labels,test_features, test_labels):
    '''
    trains and evelautes prediction of model
    '''
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions

class_corpus = final_class_data.value
class_labels = final_class_data.variable

train_corp, test_corp, train_labels, test_labels = prepare_dataset(class_corpus, class_labels, test_data_proportion=0.3)

#generate features for train and test
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_corp)
tfidf_test_features = tfidf_vectorizer.transform(test_corp)

#fit model and evaluate (used support vector machine and naive bayes)
mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter=100)

print('svm metrics')
svm_tfidf_predictions = train_predict_evaluate(
                                            classifier=svm,
                                            train_features=tfidf_train_features,
                                            train_labels=train_labels,
                                            test_features=tfidf_test_features,
                                            test_labels=test_labels
                                            )
print('\nmnb metrics')
mnb_tfidf_predictions = train_predict_evaluate(
                                            classifier=mnb,
                                            train_features=tfidf_train_features,
                                            train_labels=train_labels,
                                            test_features=tfidf_test_features,
                                            test_labels=test_labels
                                            )
