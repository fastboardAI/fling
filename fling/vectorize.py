import gensim
import matplotlib as mpl
from imp import reload
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import numpy as np
import nltk,re,pprint
import sys,glob,os
import operator, string, argparse, math, random, statistics

class vectorize:
    def __init__(self,data,factorName):
        self.data = data
        self.dataNew = []
        self.model = None
        self.swords = set(stopwords.words('english'))
        self.factorName = factorName
        for docId in range(len(self.data)):
            dv_1 = self.data[factorName][int(docId)]
            self.dataNew.append(dv_1)
        self.nDocs = len(self.dataNew)
        print(self.nDocs,"documents added!")
        
    def rem_stop_punct(self,originalText):
        splittedText = originalText.split()
        lenl = len(splittedText)
        wordFiltered = []
        tSent = []
        for r in range(lenl):
            wordx_1 = splittedText[r]
            wordx_2 = "".join(c for c in wordx_1 if c not in ('!','.',':',',','?',';','``','&','-','"','(',')','[',']','0','1','2','3','4','5','6','7','8','9')) 
            sWord = wordx_2.lower()
            if sWord not in self.swords:
                tSent.append(sWord)
        return tSent

    def tagged_document(self,list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    def trainDocVectors(self):
        self.data_for_training = list(self.tagged_document(self.dataNew))
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=30)
        self.model.build_vocab(self.data_for_training)
        self.model.train(self.data_for_training, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return(self.model)
        
    def addDocVectors(self):
        docVectors = []
        for docId in range(len(self.data)):
            docVectors.append(self.model.infer_vector(self.rem_stop_punct(self.data[self.factorName][int(docId)])))
        self.data['doc2vec'] = docVectors