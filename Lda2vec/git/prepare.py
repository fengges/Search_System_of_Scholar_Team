
import numpy as np
from tqdm import tqdm
import spacy
from gensim import corpora, models
from config import data_dir
from Lda2vec.utils import preprocess, get_windows
import sys
root=data_dir+"/lda2vec/data/mypaper/"



MIN_COUNTS = 20
MAX_COUNTS = 1800
# words with count < MIN_COUNTS
# and count > MAX_COUNTS
# will be removed

MIN_LENGTH = 15
# minimum document length
# (number of words)
# after preprocessing

# half the size of the context around a word
HALF_WINDOW_SIZE = 5
# it must be that 2*HALF_WINDOW_SIZE < MIN_LENGTH


tmp=open(root+"fenci.txt",'r',encoding='utf8').readlines()
docs =[t[1:-1] for t in  tmp]

# number of documents
print(len(docs))

# store an index with a document
docs = [(i, doc) for i, doc in enumerate(docs)]


encoded_docs, decoder, word_counts = preprocess(
    docs,  MIN_LENGTH, MIN_COUNTS, MAX_COUNTS
)


# new ids will be created for the documents.
# create a way of restoring initial ids:
doc_decoder = {i: doc_id for i, (doc_id, doc) in enumerate(encoded_docs)}


data = []
# new ids are created here
for index, (_, doc) in tqdm(enumerate(encoded_docs)):
    windows = get_windows(doc, HALF_WINDOW_SIZE)
    # index represents id of a document,
    # windows is a list of (word, window around this word),
    # where word is in the document
    data += [[index, w[0]] + w[1] for w in windows]

data = np.array(data, dtype='int64')

# a row in 'data' contains:
# id of a document, id of a word in this document, a window around this word
# 1 + 1 + 10
print(data.shape[1])

print(data.shape[0])

word_counts = np.array(word_counts)
unigram_distribution = word_counts/sum(word_counts)

vocab_size = len(decoder)
embedding_dim = 50

# train a skip-gram word2vec model
texts = [[str(j) for j in doc] for i, doc in encoded_docs]
model = models.Word2Vec(texts, size=embedding_dim, window=5, workers=4, sg=1, negative=15, iter=70)
model.init_sims(replace=True)

word_vectors = np.zeros((vocab_size, embedding_dim)).astype('float32')
for i in decoder:
    word_vectors[i] = model.wv[str(i)]

# number of unique words
print(vocab_size)

texts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

n_topics = 25
lda = models.LdaModel(corpus, alpha=0.9, id2word=dictionary, num_topics=n_topics)
corpus_lda = lda[corpus]


for i, topics in lda.show_topics(n_topics, formatted=False):
    print('topic', i, ':', ' '.join([t for t, _ in topics]))

doc_weights_init = np.zeros((len(corpus_lda), n_topics))
for i in tqdm(range(len(corpus_lda))):
    topics = corpus_lda[i]
    for j, prob in topics:
        doc_weights_init[i, j] = prob

np.save(root+'data.npy', data)
np.save(root+'word_vectors.npy', word_vectors)
np.save(root+'unigram_distribution.npy', unigram_distribution)
np.save(root+'decoder.npy', decoder)
np.save(root+'doc_decoder.npy', doc_decoder)
np.save(root+'doc_weights_init.npy', doc_weights_init)