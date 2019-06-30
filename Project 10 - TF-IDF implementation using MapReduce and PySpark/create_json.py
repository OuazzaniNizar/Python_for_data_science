import json
import os
import re 
import json

#Création du fichier JSON à partir des fichiers textes
WORD_RE = re.compile(r"[\w']+")
directory = 'C:/Users/bbbbb/Desktop/bigdat/bbc'

data = [] 
words_doc=[]
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file = open(directory+'/'+filename)
        with open(directory+'/'+filename, 'r') as f:
            num_words=0
            for line in f:
                words = line.split()
                num_words += len(words)
        words_doc.append(num_words)


        data.append({  
        'docId': filename.replace(directory,''),
        'content': " ".join(WORD_RE.findall(file.read()))
        })
        continue
    else:
        continue

print(float(sum(words_doc)/len(words_doc)))
with open('data1.json', 'w') as outfile:  
    json.dump(data, outfile)
'''
#Création des fichiers textes à partir de la base de données Excels
import xlrd 
  
loc = ("C:/Users/bbbbb/Desktop/bigdat/index.xlsx") 
  
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(1) 
  
# For row 0 and column 0 
sheet.cell_value(0, 0) 
  
print(sheet.ncols)

for j in range(400):
    try:
        f = open("C:/Users/bbbbb/Desktop/bigdat/"+str(j)+'.txt','w')
        f.write(sheet.cell_value(j,7))
        f.close()
        pass
    except Exception as e:
        continue
'''

