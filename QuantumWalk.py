"""
Continuous time random walks are an interesting problem which 
is has been shown that Quantum Computers can offer a speeded up 
solution for. Below I have written some code which could be used 
on an actual quantum computer to do this using the pyquil standard

Please not: for this to run, a qvm/quil server/client network needs to
be set up. 
"""
####################################################################
import numpy as np 
import networkx as nx 
#for dealing with physical realisation i.e which qubits can actually
#be made to talk to each other 
import matplotlib.pyplot as plt 
from scipy.linalg import expm 
#preamble 
####################################################################
#Set up physical implementation of system 

G = nx.complete_graph(8)
A = nx.adjacency_matrix(G).toarray()
#Adjacency matrices are 0,1 array based on whether two nodes are connected or not 

eigvals, _ = np.linalg.eigh(A)

ham = A + np.eye(8)
#helps simplify the quantum circuit 

#complete graphs are hadamard diagonalisable!
had = np.sqrt(1/2) * np.array([[1, 1], [1, -1]])
pauli_x = np.array([[0, 1], [1, 0]])
Q = np.kron(np.kron(had, had), had)
Q.conj().T.dot(ham).dot(Q)

#######################################################################
#now we actually get to the quantum computing part 

from pyquil import Program 
from pyquil.gates import H, X, CPHASE00
from pyquil.api import WavefunctionSimulator, local_forest_runtime
#preamble
########################################################################

wfn_sim = WavefunctionSimulator()

with local_forest_runtime():
	qvm = get_qc('9q-square-qvm')



def k_8_ctqw(t):
    #   Change to diagonal basis
    p = Program(H(0), H(1), X(0), X(1))
    
    #    Time evolve
    p += CPHASE00(-8*t, 0, 1)
    
    #   Change back to computational basis
    p += Program(X(0), X(1), H(0), H(1))
    
    return p

T = A / np.sum(A, axis=0)
time = np.linspace(0, 8, 40)
quantum_probs = np.zeros((len(time), 8))

for i, t in enumerate(time):
    p = k_8_ctqw(t)  
    wvf = wfn_sim.wavefunction(p)
    vec = wvf.amplitudes
    quantum_probs[i] = np.abs(vec)**2 
    
ax1 = plt.gca()

ax1.set_title("Quantum evolution")
ax1.set_ylabel('p')
ax1.plot(time, quantum_probs[:, 0], label='Initial node')
ax1.plot(time, quantum_probs[:, 1], label='Remaining nodes')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
