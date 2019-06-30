from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs
import time
import random
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
#import findspark
import pandas as pd
from pyspark.sql import SparkSession



sc = SparkContext.getOrCreate()


#Output format : (0, [5.4, 3.7, 1.5, 0.2, '1']) => (id, [features, 'tag'])
def loadData(filePath):
  data=sc.textFile(filePath)
  #We should only convert the coordinates to float, so we use [-1:] to escape the tag
  data = data.map(lambda x : x.split(','))\
             .zipWithIndex()\
             .filter(lambda x: x is not None)\
             .filter(lambda x: x != "")\
             .map(lambda x : (x[1],[float(coordinate) for coordinate in x[0][:-1]]+x[0][-1:]))
  return data


# Output format :
def initCentroids(k, data):
  numDataPoints = data.count()
  #Number of centroids should not excede number of data points
  if k <= numDataPoints : 
    indexCentroids = [index for index in random.sample(range(numDataPoints), k)]
    centroids=data.filter(lambda x : x[0] in indexCentroids )\
                  .map(lambda x : x[1])\
                  .zipWithIndex()\
                  .map(lambda x : (x[1],x[0]))
    return centroids 

#Output format : 
def assignToCluster(points, centroids):
  #We eliminate the tag thus using [:-1]
  pointsWithoutY = points.map(lambda x : (x[0],[float(coordinate) for coordinate in x[1][:-1]]))
  cartesianProduct = pointsWithoutY.cartesian(centroids)
  
  return cartesianProduct.map(lambda pt : (pt[0][0],pt[1][0],np.sqrt(sum([(x - y)**2 for x,y in zip(pt[0][1],pt[1][1])]))) )\
                  .map(lambda pt : (pt[0],(pt[2],pt[1])))\
                  .reduceByKey(lambda x,y : min(x,y))\
                  .map(lambda pt : (pt[0],(pt[1][1],round(pt[1][0],2))))


def computeCentroids(dataPoints,dataCentroids):
  return dataPoints.join(dataCentroids)\
                   .map(lambda x : (x[1][1][0],x[1][0][:-1]))\
                   .reduceByKey(lambda x,y : np.mean([np.array(x),np.array(y)],axis=0))\
                   .map(lambda x : (x[0], [round(coordinate,2) for coordinate in x[1]]))


def computeIntraClusterDistance(affectation):
  return affectation.map(lambda x : (x[1][0],x[1][1])).reduceByKey(lambda x,y : x+y)



def kMeans(pathData, nClusters, nIterations):
  startTime = time.time()
  dataSet = loadData(pathData)
  initialCentroids = initCentroids(nClusters,dataSet) 
  
  initialDataJoinCentroids = assignToCluster(dataSet, initialCentroids)
  centroidsAssigned = assignToCluster(dataSet, initialCentroids)
  j=round(sum([x[1] for x in computeIntraClusterDistance(centroidsAssigned).collect()]),2)
  print("Distance : ",j," - iteration :",0) 
    
  newCentroids = computeCentroids(dataSet,centroidsAssigned)
  
  for iteration in range(nIterations-1):
    nextDataJoinCentroids = assignToCluster(dataSet, newCentroids)
    j=round(sum([x[1] for x in computeIntraClusterDistance(nextDataJoinCentroids).collect()]),2)
    newCentroids = computeCentroids(dataSet,nextDataJoinCentroids)
    
    print("Distance : ",j," - iteration :",iteration+1) 

  print("Nombre iterations : ",iteration+1)
  print("Distance finale : ",round(j,2))  
  print("Centroides finaux : ", newCentroids.collect())
  timing=round((time.time() - startTime),2)
  print("%s sc" % timing)
  return (j,timing,nextDataJoinCentroids, newCentroids)


#spark-submit kmeans.py data.csv k m
if len(sys.argv) != 4:
    print("4 arguments are needed :")
    print(" * file name of the code kmeans.py")
    print(" * file name of the dataset e.g. data.csv")
    print(" * number of clusters e.g. 3")
    print(" * number of iterations e.g. 2")
    print("Try executing the following command : spark-submit kmeans.py data.csv k m")
    exit(0)

# inputs
file_name = sys.argv[1]  # file name of the dataset
nbrClusters = int(sys.argv[2]) # number of clusters
iterations = int(sys.argv[3]) # number of iteration

kMeans(file_name,nbrClusters,iterations)

