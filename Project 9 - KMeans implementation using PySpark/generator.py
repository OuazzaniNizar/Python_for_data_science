import sys
import random 
import numpy as np 
from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs
sc = SparkContext.getOrCreate()



def generator(fileName, n, k, p,s):
  mean=np.random.uniform(0,100,(k,p)) #µk tirée uniformément entre 0 et 100
  print(mean)
  standardDeviation=[s]*p
  nbrPtsPerCluster = int(n/k)
  #multiplie by mean and add sigma to get a N(mu,sigma)
  InitialCluster=RandomRDDs.normalVectorRDD(sc, numRows = n-(k-1)*nbrPtsPerCluster, numCols = p)\
                     .map(lambda x : np.array(mean[0])+np.array(x)*np.array(standardDeviation))\
                     .map(lambda x : ([round(coordinate,2) for coordinate in x],0))
  
  clusters = InitialCluster
  for numCluster in range(1,k):
    nextCluster = RandomRDDs.normalVectorRDD(sc, numRows = nbrPtsPerCluster, numCols = p)\
                            .map(lambda x : np.array(mean[numCluster])+np.array(x)*np.array(standardDeviation))\
                            .map(lambda x : ([round(coordinate,2) for coordinate in x],numCluster))
    
    clusters = clusters.union(nextCluster)

  lines = clusters.map(lambda line : ', '.join([str(x) for x in line[0]]) + ', ' + str(line[1]))
  with open(fileName,'w') as file:
    for row in lines.collect():
      file.write(row)
      file.write('\n')
  
  return lines

#spark-submit generator.py out.csv n k p s où
#— out.csv est le nom de fichier à générer
#— n est le nombre total de points à générer
#— k est le nombre de clusters
#— d est la dimensionalité des données
#— s est l’´ecart type σ.
# main code
if len(sys.argv) != 6:
    print("6 arguments are needed :")
    print(" * file name of the code generator.py")
    print(" * file name to be generated e.g. output")
    print(" * number of points to be generated e.g. 9")
    print(" * number of clusters e.g. 3")
    print(" * dimension of the data e.g. 2")
    print(" * standard deviation e.g. 10\n")
    print("Try executing the following command : spark-submit generator.py out 9 3 2 10")
    exit(0)

# inputs
file_name = sys.argv[1] + '.csv'  # file name to be generated
points = int(sys.argv[2]) # number of points to be generated
nbrClusters = int(sys.argv[3]) # number of clusters
dimension = int(sys.argv[4]) # dimension of the data
std = int(sys.argv[5]) # standard deviation

generator(file_name,points,nbrClusters,dimension,std).collect()