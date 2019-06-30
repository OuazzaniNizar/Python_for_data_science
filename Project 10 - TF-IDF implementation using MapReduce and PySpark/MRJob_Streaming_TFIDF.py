from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from mrjob.step import MRStep
from math import log
import re 
import json
import time
import numpy as np

regex = re.compile(r"[\w']+")

class MapReduceTFIDF(MRJob):
	# On précise qu'il s'agit du protocole JSON à utiliser dans la lecture
	INPUT_PROTOCOL = JSONValueProtocol
	# Sortie => (terme, docId, nbDocs), 1
	def mapper1(self, _, json_doc):
		#Longueur du document JSON ie nombre de documents/corpus
		D = len(json_doc)

		for i in range(D):
			corpus = json_doc[i]

			for terme in regex.findall(corpus['content']):
				yield (terme.lower(),corpus['docId'],D), 1
    
    # Sortie : (terme, docId, nbDocs), sum(occurence par doc)	
	def reducer1(self, termeInfos,occurence):
		yield (termeInfos[0],termeInfos[1], termeInfos[2]), sum(occurence)

	# Sortie : (docId, [terme, occurence par doc, nbDocs])	
	def mapper2(self, docInfo, n):
		yield docInfo[1], (docInfo[0],n, docInfo[2])
	# Sortie : ([terme, docId, nbDocs], [occurence terme dans docId, nombre total de terme dans docId])
	def reducer2(self,docId,termeInfo):
		total=0
		n=[]
		terme=[]
		D=[]
		for valeur in termeInfo:
			total+=valeur[1]
			n.append(valeur[1])
			terme.append(valeur[0])
			D.append(valeur[2])
		N=[total]*len(terme)

		for valeur in range(len(terme)):
			yield (terme[valeur],docId,D[valeur]),(n[valeur],N[valeur])
	
	# Sortie : (terme, [docId, occurence dans doc, nb total termes dans doc, nbDocs, 1])
	def mapper3(self, termeInfo, compteTermes):
		yield termeInfo[0], (termeInfo[1], compteTermes[0], compteTermes[1],termeInfo[2] ,1)

	# Sortie : ([terme docId, nbDocs], [occurence dans doc, nb total termes dans doc, nb docs où terme apparaît])
	def reducer3(self, terme, wordInfoComptes):
		total = 0
		docId = []
		n = []
		N = []
		D = []
		for valeur in wordInfoComptes:
			total += 1
			# id du document j
			docId.append(valeur[0])
			# nombre d'occurence du terme dans le document j
			n.append(valeur[1])
			# nombre total de termes dans le document j
			N.append(valeur[2])
			# nombre total de documents dans le corpus
			D.append(valeur[3])
        # nombre de documents où le terme ti apparaît
		m = [total]* len(n)
		for valeur in range(len(m)):
			yield (terme, docId[valeur], D[valeur]), (n[valeur], N[valeur], m[valeur])

	def mapper4(self,termeInfo,termeMetrics):
		tfidf=(termeMetrics[0]/termeMetrics[1])*np.log2(termeInfo[2]/termeMetrics[2])
		yield (termeInfo[0],termeInfo[1]),tfidf

	def steps(self):
		return [
			MRStep(mapper=self.mapper1,
				reducer=self.reducer1),
			MRStep(mapper=self.mapper2,
				reducer=self.reducer2),
			MRStep(mapper=self.mapper3,
				reducer=self.reducer3),
			MRStep(mapper=self.mapper4)]
			
if __name__ == '__main__':
	start_time = time.time()
	MapReduceTFIDF.run()
	print("--- %s seconds ---" % (time.time() - start_time))