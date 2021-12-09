import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv
from math import log10
from sklearn.linear_model import LinearRegression

wordnet = pd.read_csv('/Users/jiadesong/Desktop/ISE540/hw56/wordnet-simple.txt',sep="\t", header = None)
freebase = pd.read_csv('/Users/jiadesong/Desktop/ISE540/hw56/freebase-simple.txt',sep="\t", header = None)
# print(wordnet)
# print(freebase)
Wduniquetriple = len(wordnet.drop_duplicates())
Fruniquetriple = len(freebase.drop_duplicates())
print("Wordnet unique triples: ",Wduniquetriple)
print("Freebase unique triples: ",Fruniquetriple)

Wdunique = wordnet.drop_duplicates()
Frunique = freebase.drop_duplicates()

Wdntunique = wordnet.nunique()
print("Wordnet head unique number: ",Wdntunique[0])
print("Wordnet tail unique number: ",Wdntunique[2])
Frbsunique = freebase.nunique()
print("Freebase head unique number: ",Frbsunique[0])
print("Freebase tail unique number: ",Frbsunique[2])
Wdntuniqueentities = len((wordnet[0].append(wordnet[2])).unique())
Frbsuniqueentities = len((freebase[0].append(freebase[2])).unique())
print("Wordnet unique total entity: ",Wdntuniqueentities )
print("Freebase unique total entity: ",Frbsuniqueentities)

print("Wordnet unique relations: ",Wdntunique[1])
print("Freebase unique relations: ",Frbsunique[1])

print("The average number of Wordnet triples each relation participates in: ", Wduniquetriple/(Wdntunique[1]))
print("The average number of Freebase triples each relation participates in: ", Fruniquetriple/(Frbsunique[1]))


Wdfrequency = {}
for rela in Wdunique[1]:
   if rela in Wdfrequency:
      Wdfrequency[rela] += 1
   else:
      Wdfrequency[rela] = 1

sortedWdfrequency = dict( sorted(Wdfrequency.items(),
                           key=lambda item: item[1],
                           reverse=True))
Wdkeys = sortedWdfrequency.keys()
Wdvalues = sortedWdfrequency.values()
plt.bar(Wdkeys, Wdvalues)
plt.xticks(rotation='vertical')
plt.gca().set(title='Wordnet Frequency Histogram', ylabel='Frequency')
plt.show()

Frfrequency = {}
for relas in Frunique[1]:
   if relas in Frfrequency:
      Frfrequency[relas] += 1
   else:
      Frfrequency[relas] = 1

sortedFrfrequency = dict( sorted(Frfrequency.items(),
                           key=lambda item: item[1],
                           reverse=True))
Frkeys = sortedFrfrequency.keys()
Frvalues = sortedFrfrequency.values()
plt.bar(Frkeys, Frvalues)
plt.xticks(rotation='vertical')
plt.gca().set(title='Freebase Frequency Histogram', ylabel='Frequency')
plt.show()

plt.plot(Wdvalues,linestyle="",marker="o")
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Wordnet Ordered relations ')
plt.ylabel('Frequency')
plt.show()

plt.plot(Frvalues,linestyle="",marker="o")
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Wordnet Ordered relations ')
plt.ylabel('Freebase')
plt.show()

file = open('/Users/jiadesong/Desktop/ISE540/hw56/No-Relation-wordnet-simple.txt', "w")
writer = csv.writer(file, delimiter='\t')

for w in range(len(Wdunique)):
  writer.writerow([Wdunique[0][w], Wdunique[2][w]])

file.close()
netgraph = nx.read_edgelist('/Users/jiadesong/Desktop/ISE540/hw56/No-Relation-wordnet-simple.txt', create_using=nx.Graph(), nodetype=int)

file = open('/Users/jiadesong/Desktop/ISE540/hw56/No-Relation-Freebase-simple.txt', "w")
writer = csv.writer(file, delimiter='\t')

for w in range(len(Frunique)):
  writer.writerow([Frunique[0][w], Frunique[2][w]])

file.close()
netgraph2 = nx.read_edgelist('/Users/jiadesong/Desktop/ISE540/hw56/No-Relation-Freebase-simple.txt', create_using=nx.Graph(), nodetype=str)



def degree_distribution(G):
    DegreeFre = nx.degree_histogram(G)
    DegreeList = range(len(DegreeFre))

    NewDegreeFre = []
    NewDegreeList = []
    for i in range(len(DegreeFre)):
        if DegreeFre[i] != 0 and DegreeList[i] != 0:
            NewDegreeFre.append(DegreeFre[i])
            NewDegreeList.append(DegreeList[i])

    LogFre = []
    LogList = []
    for x in NewDegreeFre:
        LogFre.append(log10(x))
    for x in NewDegreeList:
        LogList.append(log10(x))

    LogListArr = np.array(LogList)
    LogFreArr = np.array(LogFre)
    LinearModel = LinearRegression()
    LinearModel.fit(LogListArr.reshape(-1, 1), LogFreArr)
    gamma = LinearModel.coef_
    print('Gamma is:  ')
    print(gamma)

    FittedFre = LinearModel.predict(LogListArr.reshape(-1, 1))
    Powered_y = np.power(10, FittedFre)
    Powered_x = np.power(10, LogListArr)
    plt.figure(figsize=(12, 8))
    plt.plot(Powered_x, Powered_y, 'r--')
    plt.scatter(NewDegreeList, NewDegreeFre)
    plt.loglog(base=10)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

print('For wordnet: ')
degree_distribution(netgraph)
print('Number of notes: ',netgraph.number_of_nodes())
print('Number of edges: ',netgraph.number_of_edges())
print('Average clustering coefficient: ',nx.average_clustering(netgraph))
NumTria1 = nx.algorithms.cluster.triangles(netgraph)
print("Number of triangles: ",sum(NumTria1.values()) // 3)
LargestCC = max(nx.connected_components(netgraph), key=len)
Sublargest = netgraph.subgraph(LargestCC)
print("Nodes in the largest connected component: ")
print(Sublargest.number_of_nodes())
print("Edges in the largest connected component: ")
print(Sublargest.number_of_edges())
print("Density: ", nx.density(netgraph))
print("Transitivity: ", nx.transitivity(netgraph))

print('For Freebase: ')
degree_distribution(netgraph2)
print('Number of notes: ',netgraph2.number_of_nodes())
print('Number of edges: ',netgraph2.number_of_edges())
print('Average clustering coefficient: ',nx.average_clustering(netgraph2))
NumTria2 = nx.algorithms.cluster.triangles(netgraph2)
print("Number of triangles: ",sum(NumTria2.values()) // 3)
LargestCC = max(nx.connected_components(netgraph2), key=len)
Sublargest = netgraph.subgraph(LargestCC)
print("Nodes in the largest connected component: ")
print(Sublargest.number_of_nodes())
print("Edges in the largest connected component: ")
print(Sublargest.number_of_edges())
print("Density: ", nx.density(netgraph2))
print("Transitivity: ", nx.transitivity(netgraph2))

import networkx as nx
import matplotlib.pyplot as plt
netgraph3 = nx.read_edgelist('/Users/jiadesong/Desktop/ISE540/hw56/No-Relation-wordnet-simple.txt', create_using=nx.DiGraph())
InDegree = list(netgraph3.in_degree())
OutDegree = list(netgraph3.out_degree())

InDegreeFre = []
InDegreeList = []
for i in range(len(InDegree)):
    if InDegree[i][1] != 0:
        InDegreeFre.append(InDegree[i][1])
        InDegreeList.append(i)
# print(OutDegree)

WdfreqIn = {}
for k in InDegreeFre:
   if k in WdfreqIn:
      WdfreqIn[k] += 1
   else:
      WdfreqIn[k] = 1

OutDegreeFre = []
OutDegreeList = []
for i in range(len(OutDegree)):
    if OutDegree[i][1] != 0:
        OutDegreeFre.append(OutDegree[i][1])
        OutDegreeList.append(i)
# print(OutDegree)

WdfreqOut = {}
for k in OutDegreeFre:
   if k in WdfreqOut:
      WdfreqOut[k] += 1
   else:
      WdfreqOut[k] = 1

plt.figure(figsize=(12, 8))
plt.scatter(WdfreqIn.keys(), WdfreqIn.values())
plt.loglog(base=10)
plt.xlabel('Degree')
plt.ylabel('In-Degree Frequency')
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(WdfreqOut.keys(), WdfreqOut.values())
plt.loglog(base=10)
plt.xlabel('Degree')
plt.ylabel('Out-Degree Frequency')
plt.show()
