import numpy as np
import pandas as pd
import networkx as nx

# Day1

def rca(matrix):
    # global share
    world = np.sum(matrix,0)/np.sum(matrix)
    # compare the country share with the global one
    return np.array([row/sum(row)/world for row in matrix])
    

# the routine computes and returns ECI and PCI
def Method_of_Reflections(Mcp, maximal_iterations=10):
    
    # the matrices used to project on the country and product spaces separately
    ubiquity = np.sum(Mcp, axis = 0)
    diversification = np.sum(Mcp, axis = 1)

    # initial condition (here I consider the degrees of each node)
    kc = diversification / diversification.sum()
    kp = ubiquity / ubiquity.sum()
    Mpc = np.transpose(Mcp)

    # loop
    kc_old = kc
    for iteration in range(maximal_iterations):

        kc = Mcp.dot(kp)/diversification
        kp = Mpc.dot(kc_old)/ubiquity

        kc_old = kc

    return kc/kc.mean(), kp/kp.mean()   


def ECI_PCI_Method_of_Reflections(Mcp, maximal_iterations=10):
    kc, kp = Method_of_Reflections(Mcp,maximal_iterations)
    
    eci_score = (kc - kc.mean())/kc.std()
    pci_score = (kp - kp.mean())/kp.std()
    
    return eci_score, pci_score


# this routine compute the eigenvalue ECI and PCI
def ECI_PCI_eigenvalue(Mcp):
    
    # the matrices used to project on the country and product spaces separately
    vec = np.sum(Mcp,1).astype(float)
    np.divide(np.ones_like(vec), vec, out=vec, where=vec != 0)
    Pcp = np.transpose(np.transpose(Mcp)*vec)

    vec = np.sum(Mcp,0).astype(float)
    np.divide(np.ones_like(vec), vec, out=vec, where=vec != 0)
    Ppc = np.transpose(Mcp*vec)

    # the projections
    Mcc = np.matmul(Pcp,Ppc)
    Mpp = np.matmul(Ppc,Pcp)
    
    # eci eigenvector
    eigvalues, eigvectors = np.linalg.eig(Mcc)
    eci = np.real(eigvectors[:, eigvalues.argsort()[-2]])

    # the score of the eigenvector
    eci = (eci - eci.mean())/eci.std()
    
    if np.corrcoef(eci,np.sum(Mcp,1))[0,1] < 0:
        eci *= -1

    eigvalues, eigvectors = np.linalg.eig(Mpp)
    pci = np.real(eigvectors[:, eigvalues.argsort()[-2]])

    # the score of the eigenvector
    pci = (pci - pci.mean())/pci.std()
    
        
    if np.corrcoef(pci,np.sum(Mcp,0))[0,1] < 0:
        pci *= -1
        
    return eci, pci


# the iterative FC routine returning fitness and complexity
def fitness_complexity_algorithm(Mcp, maximal_iterations = 100):

    # initial condition (here I consider the degrees of each node)
    fitness = np.array(np.sum(Mcp, axis = 1)/np.sum(Mcp))

    #complexity = np.nan_to_num(1./np.array(np.sum(Mcp, axis = 0)))
    complexity = np.array(np.sum(Mcp, axis = 0)).astype(np.float64)
    one_fit = np.ones_like(fitness)
    one_com = np.ones_like(complexity)
    np.divide(one_com, complexity, out=complexity, where=complexity != 0)
    
    complexity /= complexity.sum()
    Mpc = np.transpose(Mcp)
    inverse_fitness = np.zeros_like(fitness)

    # loop over the iterations
    for iteration in range(maximal_iterations):

        # compute the inverse fitness
        np.divide(one_fit, fitness, out=inverse_fitness, where=fitness != 0)
        
        # update the fitness
        fitness = Mcp.dot(complexity)
        fitness /= fitness.sum()
        
        # update the complexity
        complexity = Mpc.dot(inverse_fitness)
        np.divide(one_com, complexity, out=complexity, where=complexity != 0)
        complexity /= complexity.sum()

    return fitness/fitness.mean(),complexity/complexity.mean()


# Day2
def nestedness_NODF(matrix):
    deg0 = matrix.sum(1)
    deg1 = matrix.sum(0)
    dim = matrix.shape
    kmat0 = np.array([deg0 for i in range(dim[0])]).astype(np.float64)
    fill = ( kmat0.transpose() - kmat0 )
    kmat0[fill < 0] = 0
    np.divide(np.ones(kmat0.shape), kmat0, out=kmat0, where=kmat0 != 0)    
    kmat1 = np.array([deg1 for i in range(dim[1])]).astype(np.float64)
    fill = ( kmat1.transpose() - kmat1 )
    kmat1[fill < 0] = 0
    np.divide(np.ones(kmat1.shape), kmat1, out=kmat1, where=kmat1 != 0)    
    cooc0 = np.dot(matrix,matrix.transpose())
    cooc1 = np.dot(matrix.transpose(),matrix)
    cooc0 = np.multiply(cooc0,kmat0)
    cooc1 = np.multiply(cooc1,kmat1)
    norm = ( dim[0]*(dim[0]-1) + dim[1]*(dim[1]-1) )
    return ( (cooc0.sum().sum()-cooc0.diagonal().sum()) + (cooc1.sum().sum()-cooc1.diagonal().sum())) / norm
    

def simple_network_drawer(A, names, node_color, font_color, node_size, font_size, width=None, iterations=100):
    # generate graph
    G=nx.from_numpy_array(A)

    # name nodes
    mapping = dict(zip(G, names))
    G = nx.relabel_nodes(G, mapping)
    
    # nodes position
    posG = nx.spring_layout(G, iterations=iterations)

    # edges width
    if width is None:
        edge_width_by_freq = [e[2]['weight'] for e in G.edges(data=True)]
    else:
        edge_width_by_freq = width
        
    # draw 
    nx.draw_networkx(G, pos=posG, node_size=node_size, font_size=font_size, node_color=node_color,
                 font_color=font_color, width=edge_width_by_freq, edge_color='black')
                 
def Proximity_network(M, rows=False):
    if rows:
        M = M.transpose()
    Cooc = np.matmul(np.transpose(M),M)
    ubiquity = M.sum(0)
    ubiMat = np.tile(ubiquity,[M.shape[1],1])
    ubiMax = np.maximum(ubiMat,np.transpose(ubiMat)).astype(float)
    np.divide(np.ones_like(ubiMax,dtype=float), ubiMax, out=ubiMax, where=ubiMax != 0)
    Product_Space = np.multiply(Cooc,ubiMax)
    return Product_Space


def Taxonomy_Network(M, rows=False):
    if rows:
        M = M.transpose()
    diversification = M.sum(1)
    divMat = np.transpose(np.tile(diversification,[M.shape[1],1]))
    Mdiv = np.divide(M,divMat,where=divMat != 0)
    A = np.matmul(np.transpose(M),Mdiv)
    
    ubiquity = M.sum(0)
    ubiMat = np.tile(ubiquity,[M.shape[1],1])
    ubiMax = np.maximum(ubiMat,np.transpose(ubiMat)).astype(float)
    np.divide(np.ones_like(ubiMax,dtype=float), ubiMax, out=ubiMax, where=ubiMax != 0)
    Taxonomy_Network = np.multiply(A,ubiMax)
    return Taxonomy_Network


# Day3
def assist(m_0, m_1):
    d_1=m_1.sum(1)
    # let us automatically select the non-empty rows in m_1
    # and consider the summation just on them
    _m_0=m_0[d_1>0]
    _m_1=m_1[d_1>0]
    d_1=d_1[d_1>0]
    # let conside the second matrix term of the assist matrix
    mm_1=np.divide(_m_1.T, d_1).T
    
    # regarding the first term, there is the issue that if I have a product with
    # zero ubiquity, then the assist matrix will explode. In that case all
    # M_{cp} will be zero. Let's use a trick, then: let's calculate all ubiquities and
    # manually set to 1 all the 0 ones: their contribution will be still zero (due to the numerator), 
    # but we will avoid dividing by 0 in the following 
    u_0=_m_0.sum(0)
    # manually set to 1 all the 0s
    u_0[u_0==0]=1
    mm_0=np.divide(_m_0, u_0)
    return np.dot(mm_0.T,mm_1)
    
    