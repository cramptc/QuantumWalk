# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import random

#Set the size of your system
d=50
#Set the dimension of the quantum register
dim=2*d+1
#Set the number of steps
number_of_steps=50

zerozero = np.array([[1,0],[0,0]])
oneone = np.array([[0,0],[0,1]])
Idim = np.eye(dim,dim)
ort = (1/(np.sqrt(2)))
U_left= np.eye(dim,dim,1)
cU_left = np.kron(zerozero,Idim)+np.kron(oneone,U_left)
U_right= np.eye(dim,dim,-1)
cU_right = np.kron(oneone,Idim)+np.kron(zerozero,U_right)
Q = ort*np.array([[1,1],[1,-1]])
#Q = ort*np.array([[1,1j],[1j,1]])
firststep = np.kron(Q,Idim)
calcW = np.matmul(cU_right,np.matmul(cU_left,firststep))
zpro = np.array([[1,0],[0,0]])
opro = np.array([[0,0],[0,1]])
proj = [zpro,opro]


W= ort*np.block([[U_right,1j*U_right],[1j*U_left,U_left]])


def probf(a):
    return sum(np.abs(a)**2)
def prob(a):
    return np.abs(a)**2

def choose(coin):
    phead = probf(coin)
    probs = [phead,1-phead]
    probs = np.array(probs) / np.sum(probs)
    choices = [0,1]
    p = np.random.choice(choices,p=probs)
    return np.kron(proj[p],Idim)

input_state=np.zeros(dim,dtype=complex)
zstate = np.array([1,0])
onestate = np.array([0,1])
plusstate = ort*np.array([1,1])
minusstate = ort*np.array([1,-1])
input_state[d+2] = 1+0j
input_state = np.kron(onestate,input_state)



step = np.matmul(W,input_state)
#megaW = np.linalg.matrix_power(W, number_of_steps)
#psi_final = np.matmul(megaW,input_state)
for i in range(49):
    step = np.matmul(choose(step[:dim]),step)
    step = step / np.sum(step)
    step = np.matmul(step,W)
psi_final = step


x_even=np.linspace(0,dim-1,int(dim/2)+1,dtype=int)
y=[]
for x in x_even:
    y.append(prob(psi_final[x]))

print(sum(map(prob,psi_final)))
plt.plot(x_even, y)
plt.xlabel("Location d")
plt.ylabel("Output probability")
plt.title("Quantum Random Walk")
#plt.show()
plt.clf()

input_state=np.zeros(dim)
input_state[d+2] = 1
input_state = np.kron(onestate,input_state)

C= (1/2)*np.block([[U_right,U_right],[U_left,U_left]])
megaC = np.linalg.matrix_power(C, number_of_steps)
psi_final = np.matmul(megaC,input_state)

x_even=np.linspace(0,dim-1,int(dim/2)+1,dtype=int)
y_classical=[]
for x in x_even:
    y_classical.append(psi_final[x])

print(sum(psi_final))


plt.plot(x_even, y)
plt.plot(x_even, y_classical)
plt.legend(["quantum walk","classical walk"])
plt.xlabel("Location d")
plt.ylabel("Output probability")
plt.title("Classical and Quantum Random Walk")
plt.show()
