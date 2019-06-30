from pyspark import SparkContext
import re
import numpy as np
from __future__ import division
import time


sc = SparkContext.getOrCreate()

start_time = time.time()
#On lit tous les fichiers dans le répertoire courant
docs = sc.wholeTextFiles("*.txt").map(lambda t: (t[0].replace('file:/content/',''), t[1]) )

#On compte le nombre de documents
count_terms = sc.wholeTextFiles("*.txt").map(lambda t: (t[0].replace('file:/content/',''), len(t[1])) )
count_words = count_terms.collect()

#On tokenize le document d'entrée : on transforme le document en un ensemble de mots
tokenized_docs = docs.map(lambda t : (t[0], re.split("\\W+", t[1].lower()) ) )

#On compte le nombre de termes dans chaque document
term_frequency = tokenized_docs.flatMapValues(lambda x: x).countByValue()

#On compte le nombre de documents où un terme apparaît
document_frequency = tokenized_docs.flatMapValues(lambda x: x).distinct()\
                        .filter(lambda x: x[1] != '')\
                        .map(lambda t: (t[1],t[0])).countByKey()

#Fonction de calcul du score
def tf_idf(N, ct, tf, df):
    result = []

    for key, value in tf.items():
        #id du document
        doc = key[0]
        #le terme pour lequel on calcule le score
        term = key[1]
        #pour le document en question, on récupère le nombre total de termes 
        count = ct[ct[0]==doc][1]
        #pour le terme en quesiton, on récupère le nombre de documents
        #où il appparaît
        df = document_frequency[term]
        if (df>0):
          tf_idf = float(value/count)*np.log(number_of_docs/df)
        
        result.append({"doc":doc, "term":term, "score":tf_idf})
    return result
tf_idf_output = tf_idf(number_of_docs,count_words, term_frequency, document_frequency)

print("--- %s seconds ---" % (time.time() - start_time))