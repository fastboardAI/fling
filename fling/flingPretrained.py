import matplotlib as mpl
from imp import reload
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk,re,pprint
import sys,glob,os
import operator, string, argparse, math, random, statistics
import matplotlib.pyplot as plt
from sklearn import metrics

class flingPretrained:
    '''
    Trains linguistic models: doc2vec, fastText, word2vec, SDAE
    Load pretrained linguistic models: doc2vec, fastText, word2vec, SDAE
    Save group characteristics
    '''
    def __init__(self,data):
        self.data = data
        self.nDocs = len(self.data)
        self.nDocsTest = 0
        self.allDistances = {}
        self.groupedCharacteristic = {'glove' : None, 'vec_tfidf-doc2vec' : None, 'vec_tfidf-glove' : None, 'doc2vec' : None}
        self.wordVecModel = {'glove':None, 'doc2vec':None}
        print("\nDBSCAN initialized!\n")
        
    '''
    Load pretrained word vectors: gloVe, fastText, doc2vec, word2vec, SDAE
    by calling the appropriate load function for the vector type.
    '''
    def loadPretrainedWordVectors(self,vecType):
        if vecType == 'glove':
            self.wordVecModel['glove'] = self.loadGloveModel()
            print("GloVe Vectors Loaded!\n") 

    '''
    Loads the glove model provided a filename.
    TASK: edit the function to take a filename instead of hard-coding the location of the GloVe model.
    '''
    def loadGloveModel(self):
        print("Loading Glove Model\n")
        try:
            f = open('../datasets/glove.6B/glove.6B.50d.txt','r')
        except:
            f = open('datasets/glove.6B/glove.6B.50d.txt','r')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        print(len(gloveModel)," words loaded!\n")
        return(gloveModel)
    
    '''
    Returns the computed GloVe vector for the document. Note: a document contains multiple words, 
    and we have word vectors corresponding to every word in Glove
    '''
    def getDocVector(self,doc_Id):
        gvl=self.getGloveVectorList(listx)
        glove_dv = np.mean(gvl,axis=0)
        return(glove_dv)
    
    '''
    Add the new computed GloVe vector on the document to the data.
    '''
    def addDocumentGloveVector(self):
        vecL = []
        for indx in range(self.nDocs):
            listWords_1 = set(list(self.data['tfMatrix'][int(indx)]['word']))
            gvl=self.getGloveVectorList(listWords_1)
            vecL.append(np.mean(gvl,axis=0))
        self.data['glove-vector'] = vecL

    '''
    Distance between two documents using TF-IDF dictionaries.
        Method used: Using 'percentage of importance' by using tf-idf score as weights
    '''
    def distanceBtnTwoDocs(self, docId_1, docId_2):
        listWords_1 = set(list(self.data['tfMatrix'][int(docId_1)]['word']))
        listWords_2 = set(list(self.data['tfMatrix'][int(docId_2)]['word']))
        common = listWords_1.intersection(listWords_2)
        diff1_2 = listWords_1.difference(listWords_2)
        diff2_1 = listWords_2.difference(listWords_1)
        sumwt1 = self.data['sumTFIDF'][docId_1]
        sumwt2 = self.data['sumTFIDF'][docId_2]
        score_common, score_doc1, score_doc2 = 0,0,0
        #print(len(common),len(diff1_2),len(diff2_1))
        for word_c in common:
            score_1 = float(self.data['tfMatrix'][docId_1].loc[self.data['tfMatrix'][docId_1]['word'] == word_c]['tf-idf'])
            score_2 = float(self.data['tfMatrix'][docId_2].loc[self.data['tfMatrix'][docId_2]['word'] == word_c]['tf-idf'])
            score_common += abs(score_1/float(sumwt1) - score_2/float(sumwt2))
        for word_d12 in diff1_2:
            score_1 = float(self.data['tfMatrix'][docId_1].loc[self.data['tfMatrix'][docId_1]['word'] == word_d12]['tf-idf'])
            score_doc1 += score_1/float(sumwt1)
        for word_d21 in diff2_1:
            score_2 = float(self.data['tfMatrix'][docId_2].loc[self.data['tfMatrix'][docId_2]['word'] == word_d21]['tf-idf'])
            score_doc2 += score_2/float(sumwt2)
        score_total = score_common + score_doc1 + score_doc2
        return(score_total)
    
    '''
    Returns a list of GloVe vectors for all words in the document.
    '''
    def getGloveVectorList(self,listx):
        vecList = []
        nf = []
        for w in listx:
            try:
                vecList.append(self.wordVecModel['glove'][w])
            except:
                nf.append(w)
                continue        
        if len(vecList)==0:
            return([[0]*50])
        vecArray = np.stack(vecList, axis=0)
        return vecArray
    
    #document vector is the average of all the word vectors gloVe
    def getDocVector(self,listx):
        gvl=self.getGloveVectorList(listx)
        glove_dv = np.mean(gvl,axis=0)
        return(glove_dv)
    
    '''
    Returns the distance between two GloVe vectors.
    '''
    def getGloveDistance(self,docId_1,docId_2,method):
        listWords_1 = set(list(self.data['tfMatrix'].iloc[int(docId_1)]['word']))
        listWords_2 = set(list(self.data['tfMatrix'].iloc[int(docId_2)]['word']))
        if method == 'average':
            dv_1 = self.getDocVector(listWords_1)
            dv_2 = self.getDocVector(listWords_2)
            dist = np.linalg.norm(dv_1-dv_2)
            return dist
              
    def drawProgressBar(self, percent, barLen = 50):			#just a progress bar so that you dont lose patience
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i<int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()	

    '''
    sample distance between numx random documents and generate a bucketized histogram plot to get a better
    idea of the inter-document distance in the data.
    '''
    def getDistanceDistribution(self,numx,method):
        numHalf = int(numx/2)
        doca,docb = [],[]
        for i in range(numHalf):
            doca.append(random.randint(1,1026))
            docb.append(random.randint(1027,2053))
        distanceSample = []
        total = numHalf*numHalf
        for doc_1 in range(len(doca)):
            for doc_2 in range(len(docb)):
                if method == 'glove':
                    distanceSample.append(self.getGloveDistance(doca[doc_1],docb[doc_2],'average'))
                else:
                    distanceSample.append(self.getGloveDistance(doca[doc_1],docb[doc_2],'average'))
                cov = doc_1*numHalf + doc_2
                prog=(cov+1)/total
                self.drawProgressBar(prog)
        pltx = plot.hist(distanceSample,bins=20)
        return(pltx)
    
    '''
    Returns the gloVe vector for the word from the pre-trained gloVe vectors.
    '''
    def getGloveScore(self,w):
        try:
            return(self.wordVecModel['glove'][w])
        except:
            return([0*50]) 
    
    '''
    Combines document tfIDF dictionary with other document vectors to create combined vectors. 
    '''
    def doctfidf2vec(self,docId,mode):
        docVecList = []
        listWords = list(self.data['tfMatrix'][int(docId)]['word'])
        if mode == "tf-only":
            scores = list(self.data['tfMatrix'][int(docId)]['tf'])
        elif mode == "tf-idf":
            scores = list(self.data['tfMatrix'][int(docId)]['tf-idf'])
        lenW =len(listWords)
        gloveScores = [self.getGloveScore(el) for el in listWords]
        for j in range(lenW):
            temp = [float(scores[j])]*50
            #gloveScores[j]
            res = [a*b for (a,b) in zip(temp,gloveScores[j])]
            if len(res)==1:
                continue;
            else:
                docVecList.append(res)            
        return(np.mean(docVecList,axis=0))
    
    '''
    For each group in the specified column, average all the document vectors in the 
    group to create a group characteristic
    
    TASK: explore more options of averaging the vectors. '''
    def createGroupedCharacteristics(self,column):
        self.dataTrain.groupby([column])
        print("\nComputing groupCharacteristics for GloVe!")
        self.groupedCharacteristic['glove-vector'] = self.dataTrain.groupby([column])['glove-vector'].apply(np.average).to_frame()
        print("\nComputing groupCharacteristics for doc2vec!")
        self.groupedCharacteristic['doc2vec'] = self.dataTrain.groupby([column])['doc2vec'].apply(np.average).to_frame()
        #print("\nComputing groupCharacteristics for tfidf-doc2vec!")
        #self.groupedCharacteristic['vec_tfidf-doc2vec'] = self.dataTrain.groupby([column])['vec_tfidf-doc2vec'].apply(np.average).to_frame()
        print("\nComputing groupCharacteristics for tfidf-GloVe!")
        self.groupedCharacteristic['vec_tfidf-glove'] = self.dataTrain.groupby([column])['vec_tfidf-glove'].apply(np.average).to_frame()
       
    '''
    Function to return the group most simimar to the vector, based on distance computed with every group characteristics.
    '''
    def getNearestGroup(self,vec,vectorName):
        minDist = math.inf
        minGroup = None
        for colx in fdb.groupedCharacteristic[vectorName].index.values:
            vecy = fdb.groupedCharacteristic[vectorName].loc[colx].to_numpy(dtype=object)
            #distx = np.linalg.norm(vec-vecy)
            vec0 = sum([1 for x in vec if x==float(0)])
            if vec0!=50:
                distx = np.linalg.norm(scipy.spatial.distance.euclidean(vec,vecy))
                mag = np.sqrt(distx)
                print("case#1/distx",distx)
            else:
                distx = np.sqrt(np.linalg.norm(vecy))
                print("case#2/distx",distx)
            if mag<minDist:
                minDist = distx
                minGroup = colx                 
        return minGroup
    
    '''
    Explore options to optimize space using function.
    '''
    def splitTestTrain(self):
        mPt = int(self.nDocs*0.7)
        self.dataTrain = self.data[:mPt]
        self.dataTest = self.data[mPt:]
        self.nDocsTest = len(self.dataTest)
               
    '''
    Add computed group as a new column.
    '''
    def addVectorComputedGroup(self,vectorName,groupName):
        computedGroups = []
        for docId in range(self.nDocsTest):
            computedGroup = self.getNearestGroup(self.dataTest[vectorName].iloc[docId],vectorName)
            computedGroups.append(computedGroup)           
        self.dataTest[groupName] = computedGroups      
    '''
    Simple percentage count of documents which got the correct labels assigned.
    '''  
    def getAccuracy(self,compareWith,vecName):
        countCorrect = 0
        for d in range(self.nDocsTest):
            if self.dataTest[vecName].iloc[d] == self.dataTest[compareWith].iloc[d]:
                countCorrect+=1
        print("Accuracy of",vecName,countCorrect/self.nDocsTest*100,"%")
            
    '''
    Convert tfIDF dictionary for every document with precomputed word-embeddings
    '''
    def tfidf2vec(self,mode,method):
        vecL = []
        if mode == 'tf-only':
            columnName = 'vec_tf-' + method
            print("\nComputing column:",columnName)
            for indx in range(self.nDocs):
                gvl=self.doctfidf2vec(indx,'tf-only')
                vecL.append(gvl)
                prog=(indx+1)/self.nDocs
                self.drawProgressBar(prog)
        else:
            columnName = 'vec_tfidf-' + method
            print("\nComputing column:",columnName)
            for indx in range(self.nDocs):
                gvl=self.doctfidf2vec(indx,'tf-idf')
                vecL.append(gvl)
                prog=(indx+1)/self.nDocs
                self.drawProgressBar(prog)
        self.data[columnName] = vecL