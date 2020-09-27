
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
    def __init__(self,data):
        self.data = data
        self.nDocs = len(self.data)
        self.nDocsTest = 0
        self.allDistances = {}
        self.groupedCharacteristic = {'glove' : None, 'vec_tfidf-doc2vec' : None, 'vec_tfidf-glove' : None, 'doc2vec' : None}
        self.wordVecModel = {'glove':None, 'doc2vec':None}
        print("\nDBSCAN initialized!\n")
        
    def loadPretrainedWordVectors(self,vecType):
        if vecType == 'glove':
            self.wordVecModel['glove'] = self.loadGloveModel()
            print("GloVe Vectors Loaded!\n") 

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
    
    def getDocVector(self,doc_Id):
        gvl=self.getGloveVectorList(listx)
        glove_dv = np.mean(gvl,axis=0)
        return(glove_dv)
    
    def addDocumentGloveVector(self):
        vecL = []
        for indx in range(self.nDocs):
            listWords_1 = set(list(self.data['tfMatrix'][int(indx)]['word']))
            gvl=self.getGloveVectorList(listWords_1)
            vecL.append(np.mean(gvl,axis=0))
        self.data['glove-vector'] = vecL

    # distance between two documents using TF-IDF
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
    
    #get gloVe vectors for all words in the document
    def getGloveVectorList(self,listx):
        vecList = []
        nf = []
        for w in listx:
            try:
                vecList.append(self.wordVecModel['glove'][w])
            except:
                nf.append(w)
                #print(w,"not found in glove model!")
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
    
    def getGloveDistance(self,docId_1,docId_2,method):
        #listWords_1 = set(list(self.data['tfMatrix'][int(docId_1)]['word']))
        #listWords_2 = set(list(self.data['tfMatrix'][int(docId_2)]['word']))
        listWords_1 = set(list(self.data['tfMatrix'].iloc[int(docId_1)]['word']))
        listWords_2 = set(list(self.data['tfMatrix'].iloc[int(docId_2)]['word']))
        if method == 'average':
            dv_1 = self.getDocVector(listWords_1)
            dv_2 = self.getDocVector(listWords_2)
            #print("dv_1",dv_1)
            #print("dv_2",dv_2)
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

    #sample distance between n random documents 
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
    
    def getGloveScore(self,w):
        try:
            return(self.wordVecModel['glove'][w])
        except:
            return([0*50]) 
    
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
        #print([len(el) for el in docVecList])
        #vecArray = np.stack(docVecList, axis=0)
        return(np.mean(docVecList,axis=0))
    
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
       
    def getNearestGroup(self,vec,vectorName):
        minDist = math.inf
        minGroup = None
        for colx in fdb.groupedCharacteristic[vectorName].index.values:
            vecy = fdb.groupedCharacteristic[vectorName].loc[colx].to_numpy(dtype=object)
            #distx = np.linalg.norm(vec-vecy)
            is_all_zero = np.all((vecy == 0.0))
            if not is_all_zero:
                distx = np.linalg.norm(scipy.spatial.distance.euclidean(vec,vecy))
            else:
                distx = np.linalg.norm(vec)
            print(distx)
            if distx<minDist:
                minDist = distx
                minGroup = colx                 
        return minGroup
    
    def splitTestTrain(self):
        mPt = int(self.nDocs*0.7)
        self.dataTrain = self.data[:mPt]
        self.dataTest = self.data[mPt:]
        self.nDocsTest = len(self.dataTest)
               
    def addVectorComputedGroup(self,vectorName,groupName):
        computedGroups = []
        for docId in range(self.nDocsTest):
            computedGroup = self.getNearestGroup(self.dataTest[vectorName].iloc[docId],vectorName)
            computedGroups.append(computedGroup)           
        self.dataTest[groupName] = computedGroups
        
        
    def getAccuracy(self,compareWith,vecName):
        countCorrect = 0
        for d in range(self.nDocsTest):
            if self.dataTest[vecName].iloc[d] == self.dataTest[compareWith].iloc[d]:
                countCorrect+=1
        print("Accuracy of",vecName,countCorrect/self.nDocsTest*100,"%")
            
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
        print("\nTraining doc2vec model.")
        self.data_for_training = list(self.tagged_document(self.dataNew))
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=30)
        self.model.build_vocab(self.data_for_training)
        self.model.train(self.data_for_training, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return(self.model)
        
    def addDocVectors(self):
        print("\nAdding doc2vec vectors to dataset.")
        docVectors = []
        for docId in range(len(self.data)):
            docVectors.append(self.model.infer_vector(self.rem_stop_punct(self.data[self.factorName][int(docId)])))
        self.data['doc2vec'] = docVectors