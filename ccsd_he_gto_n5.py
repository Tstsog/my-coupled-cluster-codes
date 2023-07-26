# This python code computes  the ground state energy for helium atom from the Coupled Cluster Singles and Doubles (CCSD) method
# using the Gaussian-type orbital (GTO) with n=5 (5 s-function)
#
# The self-consistent field (SCF) calculation was carried out using the UNDMOL
# computational chemistry package at University of North Dakota; the
# two-electron integral (He_tei_data_gto_n2.txt) data was obtained from the SCF calculation.
#
# Refs: John F. Stanton, Jorgen Gauss, John D. Watts, and Rodney J.
# Bartlett, J. Chem. Phys. 94, 4334–4345 (1991);
#
#% Written by Tsogbayar Tsednee (PhD)
#% An original CCSD python code can be found at https://joshuagoings.com/2013/07/17/coupled-cluster-with-singles-and-doubles-ccsd-in-python/
#%
#% Email: tsog215@gmail.com
#% July 13, 2023 & University of North Dakota

from __future__ import division
import math
import numpy as np

####################################
#
#   FUNCTIONS
#
####################################


####################################################

dim1B = 10

#1st state
holes = [9, 8]
particles = [7, 6, 5, 4, 3, 2, 1, 0]
#
Eorb = [171.602444, 26.364991, 5.186112 , 0.814206, -0.916869] # molecular orbital energies
EN   = -2.859894932681

###
fs = np.zeros(dim1B)
for i in range(0, dim1B):
    fs[i] = 1.*Eorb[i//2]
fs = np.diag(fs)
#print('fs = ', fs)

Nbasis = 5
tei_data = np.zeros((Nbasis, Nbasis, Nbasis, Nbasis))
          #
# print(tei_data)

for line in open('He_tei_data_gto_n5.txt'):
  p, q, r, s, v = line.split()
  p = int(p) - 1
  q = int(q) - 1
  r = int(r) - 1
  s = int(s) - 1
  val = float(v)

  #[print(val)]
  tei_data[p,q,r,s] = val
  tei_data[q,p,r,s] = val
  tei_data[p,q,s,r] = val
  tei_data[q,p,s,r] = val
  tei_data[r,s,p,q] = val
  tei_data[s,r,p,q] = val
  tei_data[r,s,q,p] = val
  tei_data[s,r,q,p] = val

#  print('tei_data', tei_data)

teimo = tei_data 
# This makes the spin basis double bar integral (physicists' notation)

dim = Nbasis
spinints=np.zeros((dim*2,dim*2,dim*2,dim*2))
for p in range(0,dim*2+0):
  for q in range(0,dim*2+0):
    for r in range(0,dim*2+0):
      for s in range(0,dim*2+0):
        value1 = teimo[(p+0)//2,(r+0)//2,(q+0)//2,(s+0)//2] * (p%2 == r%2) * (q%2 == s%2)
        value2 = teimo[(p+0)//2,(s+0)//2,(q+0)//2,(r+0)//2] * (p%2 == s%2) * (q%2 == r%2)
        spinints[p-0,q-0,r-0,s-0] = value1 - value2

#####################################################
#
#  Spin basis fock matrix eigenvalues 
#
#####################################################

fs = np.zeros((dim*2))
for i in range(0,dim*2):
    fs[i] = Eorb[i//2]
fs = np.diag(fs) # put MO energies in diagonal array

#print('fs = ', fs)
#######################################################
#
#   CCSD CALCULATION
#
#######################################################
dim = dim*2 # twice the dimension of spatial orbital

# Init empty T1 (ts) and T2 (td) arrays

ts = np.zeros((dim,dim))
td = np.zeros((dim,dim,dim,dim))

# Initial guess T2 --- from MP2 calculation!

for a in particles:
  for b in particles:
    for i in holes:
      for j in holes:
        td[a,b,i,j] += spinints[i,j,a,b]/(fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b])

# Make denominator arrays Dai, Dabij
# Equation (12) of Stanton
Dai = np.zeros((dim,dim))
for a in particles:
  for i in holes:
    Dai[a,i] = fs[i,i] - fs[a,a]

# Stanton eq (13)
Dabij = np.zeros((dim,dim,dim,dim))
for a in particles:
  for b in particles:
    for i in holes:
      for j in holes:
        Dabij[a,b,i,j] = fs[i,i] + fs[j,j] - fs[a,a] - fs[b,b]

# Stanton eq (9)
def taus(a,b,i,j):
  taus = td[a,b,i,j] + 0.5*(ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j])
  return taus

# Stanton eq (10)
def tau(a,b,i,j):
  tau = td[a,b,i,j] + ts[a,i]*ts[b,j] - ts[b,i]*ts[a,j]
  return tau

# We need to update our intermediates at the beginning, and 
# at the end of each iteration. Each iteration provides a new
# guess at the amplitudes T1 (ts) and T2 (td), that *hopefully*
# converges to a stable, ground-state, solution.

def updateintermediates(x):
  if x == True:
    # Stanton eq (3)
    Fae = np.zeros((dim,dim))
    for a in particles:
      for e in particles:
        Fae[a,e] = (1 - (a == e))*fs[a,e]
        for m in holes:
          Fae[a,e] += -0.5*fs[m,e]*ts[a,m]
          for f in particles:
            Fae[a,e] += ts[f,m]*spinints[m,a,f,e] 
            for n in holes:
              Fae[a,e] += -0.5*taus(a,f,m,n)*spinints[m,n,e,f]

    # Stanton eq (4)
    Fmi = np.zeros((dim,dim))
    for m in holes:
      for i in holes:
        Fmi[m,i] = (1 - (m == i))*fs[m,i]
        for e in particles:
          Fmi[m,i] += 0.5*ts[e,i]*fs[m,e]
          for n in holes:
            Fmi[m,i] += ts[e,n]*spinints[m,n,i,e] 
            for f in particles:
              Fmi[m,i] += 0.5*taus(e,f,i,n)*spinints[m,n,e,f]

    # Stanton eq (5)
    Fme = np.zeros((dim,dim))
    for m in holes:
      for e in particles:
        Fme[m,e] = fs[m,e]
        for n in holes:
          for f in particles:
            Fme[m,e] += ts[f,n]*spinints[m,n,e,f]

    # Stanton eq (6)
    Wmnij = np.zeros((dim,dim,dim,dim))
    for m in holes:
      for n in holes:
        for i in holes:
          for j in holes:
            Wmnij[m,n,i,j] = spinints[m,n,i,j]
            for e in particles:
              Wmnij[m,n,i,j] += ts[e,j]*spinints[m,n,i,e] - ts[e,i]*spinints[m,n,j,e]
              for f in particles:
                Wmnij[m,n,i,j] += 0.25*tau(e,f,i,j)*spinints[m,n,e,f]

    # Stanton eq (7)
    Wabef = np.zeros((dim,dim,dim,dim))
    for a in particles:
      for b in particles:
        for e in particles:
          for f in particles:
            Wabef[a,b,e,f] = spinints[a,b,e,f]
            for m in holes:
              Wabef[a,b,e,f] += -ts[b,m]*spinints[a,m,e,f] + ts[a,m]*spinints[b,m,e,f]
              for n in holes:
                Wabef[a,b,e,f] += 0.25*tau(a,b,m,n)*spinints[m,n,e,f]

    # Stanton eq (8)
    Wmbej = np.zeros((dim,dim,dim,dim))
    for m in holes:
      for b in particles:
        for e in particles:
          for j in holes:
            Wmbej[m,b,e,j] = spinints[m,b,e,j]
            for f in particles:
              Wmbej[m,b,e,j] += ts[f,j]*spinints[m,b,e,f]
            for n in holes:
              Wmbej[m,b,e,j] += -ts[b,n]*spinints[m,n,e,j]
              for f in particles:
                Wmbej[m,b,e,j] += -(0.5*td[f,b,j,n] + ts[f,j]*ts[b,n])*spinints[m,n,e,f]

    return Fae, Fmi, Fme, Wmnij, Wabef, Wmbej

# makeT1 and makeT2, as they imply, construct the actual amplitudes necessary for computing
# the CCSD energy (or computing an EOM-CCSD Hamiltonian, etc)

# Stanton eq (1)
def makeT1(x,ts,td):
  if x == True:
    tsnew = np.zeros((dim,dim))
    for a in particles:
      for i in holes:
        tsnew[a,i] = fs[i,a]
        for e in particles:
          tsnew[a,i] += ts[e,i]*Fae[a,e]
        for m in holes:
          tsnew[a,i] += -ts[a,m]*Fmi[m,i]
          for e in particles:
            tsnew[a,i] += td[a,e,i,m]*Fme[m,e]
            for f in particles:
              tsnew[a,i] += -0.5*td[e,f,i,m]*spinints[m,a,e,f]
            for n in holes:
              tsnew[a,i] += -0.5*td[a,e,m,n]*spinints[n,m,e,i]
        for n in holes:
          for f in particles: 
            tsnew[a,i] += -ts[f,n]*spinints[n,a,i,f]
        tsnew[a,i] = tsnew[a,i]/Dai[a,i]
  return tsnew

# Stanton eq (2)
def makeT2(x,ts,td):
  if x == True:
    tdnew = np.zeros((dim,dim,dim,dim))
    for a in particles:
      for b in particles:
        for i in holes:
          for j in holes:
            tdnew[a,b,i,j] += spinints[i,j,a,b]
            for e in particles:
              tdnew[a,b,i,j] += td[a,e,i,j]*Fae[b,e] - td[b,e,i,j]*Fae[a,e]
              for m in holes:
                tdnew[a,b,i,j] += -0.5*td[a,e,i,j]*ts[b,m]*Fme[m,e] + 0.5*td[b,e,i,j]*ts[a,m]*Fme[m,e]
                continue
            for m in holes:
              tdnew[a,b,i,j] += -td[a,b,i,m]*Fmi[m,j] + td[a,b,j,m]*Fmi[m,i]
              for e in particles:
                tdnew[a,b,i,j] += -0.5*td[a,b,i,m]*ts[e,j]*Fme[m,e] + 0.5*td[a,b,j,m]*ts[e,i]*Fme[m,e]
                continue
            for e in particles:
              tdnew[a,b,i,j] += ts[e,i]*spinints[a,b,e,j] - ts[e,j]*spinints[a,b,e,i]
              for f in particles:
                tdnew[a,b,i,j] += 0.5*tau(e,f,i,j)*Wabef[a,b,e,f]
                continue
            for m in holes:
              tdnew[a,b,i,j] += -ts[a,m]*spinints[m,b,i,j] + ts[b,m]*spinints[m,a,i,j]  
              for e in particles:
                tdnew[a,b,i,j] +=  td[a,e,i,m]*Wmbej[m,b,e,j] - ts[e,i]*ts[a,m]*spinints[m,b,e,j]
                tdnew[a,b,i,j] += -td[a,e,j,m]*Wmbej[m,b,e,i] + ts[e,j]*ts[a,m]*spinints[m,b,e,i]
                tdnew[a,b,i,j] += -td[b,e,i,m]*Wmbej[m,a,e,j] + ts[e,i]*ts[b,m]*spinints[m,a,e,j]
                tdnew[a,b,i,j] +=  td[b,e,j,m]*Wmbej[m,a,e,i] - ts[e,j]*ts[b,m]*spinints[m,a,e,i]
                continue
              for n in holes:
                tdnew[a,b,i,j] += 0.5*tau(a,b,m,n)*Wmnij[m,n,i,j]
                continue
            tdnew[a,b,i,j] = tdnew[a,b,i,j]/Dabij[a,b,i,j] 
    return tdnew

# Expression from Crawford, Schaefer (2000) 
# DOI: 10.1002/9780470125915.ch2
# Equation (134) and (173)
# computes CCSD energy given T1 and T2
def ccsdenergy():
  ECCSD = 0.0
  for i in holes:
    for a in particles:
      ECCSD += fs[i,a]*ts[a,i]
      for j in holes:
        for b in particles:
          ECCSD += 0.25*spinints[i,j,a,b]*td[a,b,i,j] + 0.5*spinints[i,j,a,b]*(ts[a,i])*(ts[b,j]) 
  return ECCSD

#================
# MAIN LOOP
# CCSD iteration
#================
ECCSD = 0
DECC = 1.0
while DECC > 0.000000001: # arbitrary convergence criteria
  OLDCC = ECCSD
  Fae,Fmi,Fme,Wmnij,Wabef,Wmbej = updateintermediates(True)
  tsnew = makeT1(True,ts,td)
  tdnew = makeT2(True,ts,td)
  ts = tsnew
  td = tdnew
  ECCSD = ccsdenergy()
  DECC = abs(ECCSD - OLDCC)
  
  print("E(corr,CCSD) = ", ECCSD)
  print("E(CCSD) = ", ECCSD + EN)




