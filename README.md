# fLing : Fast Linguistics 
Library for all your unsupervised linguistics tasks.
![fling](fling.png)

###Introduction
This is a library for performing unsupervised lingustic functionalities based on textual fields on your data. An API will also be released for real-time inference. This is a small part of project fling, which is an opensource linguistic library designed for easy integration to applications. 

Primary functionalities
- Pre-process text columns in dataset, with custom tokenizers.
- Add tf-idf information as a new column to a dataset.
- Add pretrained word embeddings to convert raw text to document embeddings (word2vec, glove, fastText, custom).
- Use tfidf2vec module to convert tfidf information into vectors/embeddings.   
- Train tf-idf word vectors on full training dataset on categories for fast clustering inference. 
- Apply DBSCAN based on one or combined embeddings created in the previous steps, along with original information in the dataset. 
- Save cluster characteristic file and generate clusters on a new data. 
- Use clusterID's as a new feature for other supervised and unsupervised tasks. 
- More functionalities will be added. 
 
*fastboardAI/fling*
https://github.com/fastboardAI/fling.git

Latest Developments tracked in
*arnab64/fling*
https://github.com/arnab64/fling.git

Usage
-------
Basic usage instructions. As the code is in development, it might not be stable.  More details will be added by 09/30/2020 for proper usage of the library.

*Reading data*
```python
from fling import textclustering
from fling import embeddings
from fling import dbscan
```
For now, operations are performed in Pandas dataframes, and the file format we read is csv.
```python
#change operating folder      
os.chdir("/Users/arnabborah/Documents/repositories/textclusteringDBSCAN/scripts/")

#read the .csv data file using the dataProcessor class
rp = tfm.dataProcessor("../datasets/DataAnalyst.csv")
```

### using the generic TF-IDF module (unsupervised)
```python
#create a flingTFIDF object around the pre-processed daa
ftf = tfm.flingTFIDF(rp.dataInitialSmall,'Job Description')

# tokenization, customizable
ftf.smartTokenizeColumn()

# get Term Frequency of each document, and store add it as an object, in a new column
ftf.getTF()

# compute Inverse Document Frequencies across the entire vocabulary
ftf.computeIDFmatrix()

# get TFIDF, and store it as a new column in data, tf-idf
ftf.getTFIDF()

# compute sum of all tf-idf values and add it as a new column
ftf.createDistanceMetadata()
ftf.writeToFile()
```

### using the categeorical TF-IDF module (semi-supervised)
```python
from textclustering import categoricalCharacteristicModule as ccm

rp = dataProcessor("../datasets/DataAnalyst.csv")

# performing custom categorical operations on the data-frame
rp.customProcessData()

fcat = flingCategoricalTFIDF()
allfnames = ft.getallfilenames("/Users/arnabborah/Documents/repositories/textclusteringDBSCAN/processFiles/trainCatFiles")
ft.computeTFIDFallfiles(allfnames)
```

### adding pre-trained GloVe vectors
```python
from textclustering import flingPretrained as pre

dataProcessed = pd.read_pickle('../processFiles/data_tfidf_processed.pkl')
fdb = pre.flingPretrained(dataProcessed)
fdb.loadPretrainedWordVectors('glove')
fdb.addDocumentGloveVector()

# to get a sample of the distance distribution, where the first param is number of random documents 
fdb.getDistanceDistribution(200,'glove')
fdb.getDistanceDistribution(500,'glove')
```

### tfidf2vec : convert tf-idf information into vectors using pre-trained word vectors (GloVe)
```python
# converting tf-idf to vector using term frequencies information only
fdb.tfidf2vec('tf-only')
# converting tf-idf to vector using tf-idf information 
fdb.tfidf2vec('tf-idf')
```
