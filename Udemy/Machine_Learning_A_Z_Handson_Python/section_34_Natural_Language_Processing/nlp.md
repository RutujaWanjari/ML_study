## Types of NLP


#### NLP algos

1. If else rule - mechanical approach, main ly used in building **Chatbots** (bank), quickly becomes mess when multiple nested categories emerge
2. Audio Frequency component analysis - mathematical approach to match pre analysed frequencies of word to new frequencies of words, used in **Speech Recognition**
3. Bag of Words - putting all phrases with their result (variable y) in a bag, and checking which word is more associated to certain result, mainly used for classiffication problems like movie review or student grading system

#### DL algos

#### DNLP algos

1. CNN for text recognition - classification model, applying embedding(converting data to matrix), convolution(getting data square wise), pooling(pooling square data into a vector), flattening, etc

#### Sequence to Sequence Algo

1. Encoder and Decoder of the data


## NLP Pipeline

1. [Stemming vs lemmatization](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)
2. Stemmming and lemmatization both gives base form of a word, although stemming might not give you actual english word, although lemmatization gives actual english word
3. Stemming is faster as it applies an algo to get base form of all words by slicing the words, Used where speed is more prefered, like **review system**
4. Lemmatization is slower than stemming, as it check the word should be actual english word, hence traverses through whole corpus of data. This can be used where language is more preffered like **google translate**
5. **NLP cleaning pipeline**

   1. regular expression to remove all puntuations
   2. lower all words
   3. split into list
   4. apply stemming to words not present in stopwords
   5. Rejoin new list of words into single sentence and add to main corpus
   6. Apply tokenization
   7.
