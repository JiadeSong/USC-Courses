import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import log10
from sklearn.linear_model import LinearRegression
import random
"""
MAX POSSIBLE POINTS: 100
*Total possible points may add to more than 100 (extra credit).

For  this homework, you will need to have the networkx package installed. If you are using pycharm, you can install
this package just like you installed numpy, sklearn etc. earlier. If not using pycharm, use pip to install the package in your
interpreter. For more details, see  networkx's documentation on  installation. It is usually straightforward.
You will know you have succeeded when you do not see an error in  the first line (import networkx as nx).

NetworkX webpage: https://networkx.github.io/

As its name suggests, networkx is a 'graph theory' package that you can use to read, write and manipulate
networks of all kinds, including directed, undirected, weighted and attributed networks. It also gives you
many of the tools you'll ever need for getting diagnostic info on a network, including degree distribution,
connected components, and even community detection. I suggest looking at the resources in the tutorial section
to familiarize yourself with the basics: https://networkx.github.io/documentation/latest/tutorial.html
"""

"""
The network that we will be dealing with in this assignment is openly available in Stanford's SNAP ecosystem 
and involves Facebook data: http://snap.stanford.edu/data/ego-Facebook.html

"""

"""
As always, the first step is to download and familiarize yourself with the data. There should be three files at the
bottom of the html page I gave  you the link above for. The network we will be dealing with is facebook_combined.txt.gz,
which you should decompress to get the txt file. Note that sometimes
you have to refresh the page or try a different browser if you don't see the links to the files at the bottom. If none
of those work and you still don't see the files, write to me. 
"""

"""
The dataset that you see is what's called an edge-list. Each line is an edge e.g., 0 10 means that node 0 is linked to
node 10. Numerical IDs have been assigned to nodes for anonymization purposes. In this case, we will treat the network as undirected.
(WCC and SCC on the webpage stand for weakly connected components and strongly connected components. The metrics
are identical for them, as they always are if the network is undirected, since there is no distinction between
weakly connected/strongly connected components, which is why we only use the term 'connected components' in
undirected networks).  
"""

def read_in_network(input_path):
    """
    [5  points] In this function, which could be very short, you should read in  the facebook text file as a
    networkX graph object. See what I said above about the graph being undirected. Hint: there is a very nice
    and elegant way in networkX to read edge lists directly from file into a graph object.
    :param input_path: the path to the decompressed facebook_combined txt file.
    :return: a networkX graph object G
    """
    netgraph = nx.read_edgelist(input_path, create_using=nx.Graph(), nodetype=int)
    print(nx.info(netgraph))
    nx.draw(netgraph, with_labels=True)
    plt.show()
    return netgraph

kk = read_in_network("/Users/jiadesong/Desktop/ISE540/hw4/facebook_combined.txt")

def verify_webpage_numbers(G):
    """
    [25 points] In this function, write code (you can use networkX functionality to any extent that you wish) to verify
    the numbers on the SNAP webpage describing this dataset regarding the average clustering coefficient, the number of
    triangles, the diameter, and the nodes and edges in the largest connected component. Is there any discrepancy?
    Please report if so.
    :param G: the networkX graph object
    :return: None
    """
    ACC = nx.average_clustering(G)
    print("Average clustering coefficient: ")
    print(ACC)

    NumTria = nx.algorithms.cluster.triangles(G)
    print("Number of triangles: ")
    print(sum(NumTria.values())//3)

    diam = nx.diameter(G)
    print("Diameter: ")
    print(diam)

    LargestCC = max(nx.connected_components(G), key=len)
    Sublargest = G.subgraph(LargestCC)
    print("Nodes in the largest connected component: ")
    print(Sublargest.number_of_nodes())
    print("Edges in the largest connected component: ")
    print(Sublargest.number_of_edges())


verify_webpage_numbers(kk)
'''
According to SNAP Website: 
Average clustering coefficient = 0.6055
Number of triangles	= 1612010
Diameter (longest shortest path) = 8
Nodes in largest CC = 4039 (1.000)
Edges in largest CC = 88234 (1.000)
'''

'''
Answer: 
After Calculation, there is no discrepancy. 
'''
def degree_distribution(G):
    """
    [15 points] Plot the degree distribution on a log-log plot. You can use python (e.g., matplotlib and pyplot if you
    wish for the plot, or collect data  and plot it somewhere else, like excel).
    Does the degree distribution obey the power law well? If so, what is the power-law coefficient? (Recall that a power
    law says that the prob(k) of degree k is proportional to k^{-gamma}. Gamma is known as  the power-law coefficient
    (it also has other names).

    The plot should be separately included with  your homework submission (a short PDF report, containing answers/figures,
    with captions to any questions that cannot be directly answered in file here)
    
    Hint: On a log-log plot, the power-law is linear. You can use linear regression (such as in sklearn) to deduce gamma.
    
    :param G: the networkX graph object
    :return: None
    """
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

    """
    The power-law coefficient is Gamma = -1.18. 
    """
# degree_distribution(kk)

    #[5 points] How many friends does the node with the highest degree have? What is the clustering coefficient
    # of that node? Interpret the result: do you believe this individual is a member of a cohesive and tightly knit
    # group  or more like a central figure that people are connecting to for some reason (e.g., celebrity status
    # or because the person is popular, or just adding lots of friends on facebook for no reason, as some people do)

max_degree = max(kk.degree, key=lambda x: x[1])[1]
Maxnode = [node[0] for node in kk.degree if node[1] == max_degree]
print("Node name with clustering coefficient: ")
print(nx.clustering(kk, Maxnode))
print('Number of friends the node with the highest degree have:')
print(max_degree)
"""
Answer: 
The node of highest number of friends: 107, the clustering coefficient is 0.0490, the number of friends is 1045. 
"""
    #[10 points extra credit]. Separately report the degree distributions of the ten individual networks on the dataset
    #webpage under file facebook.tar.gz. The folder (obtained upon decompressing) contains many files; edge lists are contained in [num].edges files.




"""
Now, we will analyze what happens to the metrics if you only get to observe a 'fraction' of the network, as most of us do when
we are working  with these kinds of networks at very large-scale. 
"""

def sample_edges(G, num_edges):
    """
    [10 points] Complete the code. Make sure to use numpy or scipy for your sampling needs, as always. Note that because
    we are sampling edges, rather than nodes, nothing can be said about the number of nodes in the output graph. In theory,
    because there are roughly 4000+ nodes, you may end up getting all nodes even if you sample only 5000 edges!
    :param G: the original graph
    :param num_edges: number of edges to sample. If -1 or greater than the total number of edges in G, just
    return the full graph G.
    :return: another graph H that contains num_edges randomly sampled edges from G
    """
    if num_edges < 0 or num_edges > G.number_of_edges():
        return G
    EdgeList = list(G.edges)
    random_edges = random.sample(EdgeList, num_edges)
    H = G.edge_subgraph(random_edges)
    return H

"""
[40 points]
Use  the function above  (including any others you need) to randomly sample edges from G to produce four graphs with 15000, 30000, 45000 and 60000 edges respectively.
Now re-compute the  average clustering coefficient, fraction of closed triangles and diameter (note that
if the graph is disconnected, the diameter is infinite. Please note if this happens) on these graphs. For
best results, you should sample (for each size) ten times and then compute averages and standard deviations of the network metrics noted above. Ignore infinite diameters
for the purpose of averaging, unless the majority of diameters in  your 10-sample are infinite. 

In the report, tabulate these results and provide a succinct conclusion. Is there a sample size at which the
network starts looking like the original network? Which (if any) of the quantities you have computed above
are reliable and at what sampling threshold? For example, you may conclude, 'fraction of closed triangles starts
approaching the level in  the original network even in the 15000 edge sample, but the diameter continues to be
untrustworthy till we see at least 45000 edges...'. 

I am looking for short, insightful reports. Statistical significance testing, if applied will boost your score. 
In particular, look to such testing for writing a decisive conclusion.   
"""
def recompute(G):

    for i in [15000, 30000, 45000, 60000]:
        print("Sample size: ")
        print(i)
        ACC = []
        NumTria = []
        Diam = []
        DiamInfiNum = 0
        for j in range(10):
            # print('The number of sample: ')
            # print(j+1)
            H = sample_edges(G, i)
            ACC.append(nx.average_clustering(H))
            NumTria1 = nx.algorithms.cluster.triangles(H)
            NumTria2 = sum(NumTria1.values()) // 3
            NumTria.append(NumTria2)
            if DiamInfiNum <= 5:
                if nx.is_connected(H):
                    Diam.append(nx.diameter(H))
                    print(j + 1)
                else:
                    print('H is disconnected and diameter is infinite. ')
                    print(j + 1)
                    DiamInfiNum = DiamInfiNum + 1

        print("Average and STD ACC: ")
        print(np.average(np.array(ACC)))
        print(np.std(np.array(ACC)))
        print('Average and STD Number of Triangles: ')
        print(np.average(np.array(NumTria)))
        print(np.std(np.array(NumTria)))
        if DiamInfiNum <= 5:
            print('Average and STD diameter: ')
            print(np.average(np.array(Diam)))
            print(np.std(np.array(Diam)))
        else:
            print('The majority of diameters in 10-sample are infinite')

# recompute(kk)
"""
[10 points]. You are trying to sample x edges, and the metric of success is the degree distribution
(both in shape and if the power-law applies, the power-law coefficient) and its similarity for your sampled network
to the original network. Rounded to the nearest 5000, at what point does your sample bring you 'close enough' to the original (i.e. what is the value of x we should use)?
We are qualitatively unwilling to accept an 'error' of more than 10% e.g., the true coefficient must be within 10% of
the coefficient you get from your sampled network's degree distribution. 
"""
def Sample_X_Edges(G):
    for i in range(1,20):
        SampleSize = 5000*i
        H = sample_edges(G, SampleSize)
        frequency = nx.degree_histogram(H)
        DegreeList = range(len(frequency))

        NewDegreeFre = []
        NewDegreeList = []
        for j in range(len(frequency)):
            if frequency[j] != 0 and DegreeList[j] != 0:
                NewDegreeFre.append(frequency[j])
                NewDegreeList.append(DegreeList[j])

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
        print('Sample size: ')
        print(5000*i)
        print('Gamma is:  ')
        print(gamma)
        error = abs((gamma - (-1.18016442)) / (-1.18016442))
        if error < 0.1:
            print (SampleSize,error)
            break
# good luck!
"""
When sample size is more than 50000, the error is 0.09805157 < 10%. 
"""
Sample_X_Edges(kk)