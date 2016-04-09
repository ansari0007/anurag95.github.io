#import dataset
from reviews import *

import gensim
from random import shuffle
import nltk
import numpy as np
#shuffle(tagged_reviews)

size = 100000

tagged_reviews = tagged_reviews[:size]
sentences = []
for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	if len(review):
		sentences.append(review)

model = gensim.models.Word2Vec(sentences, min_count=1)

final = []

for label, review in tagged_reviews:
	review = nltk.word_tokenize(review)
	review_vector = np.array([0.0 for i in range(100)])
	if not len(review):
		continue
	for word in review:
		review_vector += model[word]
	review_vector = review_vector/len(review)
	final.append((label, list(review_vector)))

print "Final ban gya"

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from sklearn.externals import joblib


alldata = ClassificationDataSet(len(final[0][1]), 1, nb_classes=2)
for i, tup in enumerate(final):
    alldata.addSample(tup[1], tup[0])

tstdata, trndata = alldata.splitWithProportion(0.25)

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
trainer.trainUntilConvergence(maxEpochs = 30)

joblib.dump(fnn, "fnn_model.pkl")