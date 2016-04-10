import numpy as np
import matplotlib.pyplot as plt
from NRMethod import Newton_BT

# VAR GLOBALES :
global n
global alpha
global L
global epsilon
global DELTA_t
n=100
alpha=float(3)
L=float(8)
epsilon=0.1
DELTA_t=0.1

#FUNCTIONS :

def ComputeU0():
    """Function COmputeu0 computes U0 (vector of size n) on the interval [-L,L]. Init cond: zheta(x,0) = exp(-alpha*x^2) """
    zheta = lambda x, alpha : np.exp(-alpha*(x**2))
    U0 = np.zeros([n,1])
    print(U0)
    for i in range(n):
        print(zheta((float(2*i-n+1)/(n-1))*L,alpha))
        U0[i][0] = zheta((float(2*i-n+1)/(n-1))*L,alpha)
    return U0

def Draw(Uk):
    """ Function Draw draws Uk of size n on the interval [-L,L]"""    
    vecx = np.zeros([n,1])
    for i in range(n):
        vecx[i][0] =(float(2*i-n+1)/(n-1))*L
    plt.plot(vecx, Uk, linewidth=1.0)
    plt.show()

def G(U):
    """Function G with parameter DELTA_x computed in function of L,
    used in the wave equation """
    n = U.shape[0]
    G_U = np.zeros([n,1])
    DELTA_x = float(2*L)/(n-1)
    for i in range(n):
        G_U[i][0] = U[(i+1)%n][0]
        G_U[i][0] -= U[(i-1)%n][0]
        G_U[i][0] /= (2* DELTA_x)
        G_U[i][0] += (float(epsilon) * (U[(i+1)%n][0]- U[(i-1)%n][0]) * (U[(i-1)%n][0]+U[(i+1)%n][0]+ U[i][0])/ (4* DELTA_x))
        G_U[i][0] += (float(epsilon) * (U[(i+2)%n][0]- 2*U[(i+1)%n][0]+ 2*U[(i-1)%n][0]- U[(i-2)%n][0]) / (12*( DELTA_x**3)))
    return G_U 

def H_G(X):
    """ Function calculating the jacobian matrix of the function G returned by G """
    n = X.shape[0]
    H = np.zeros([n,n])
    DELTA_x = float(2*L)/(n-1)
    for i in range(n):
        H[i][(i-2)%n] = - float(epsilon) / (12 * DELTA_x)
        H[i][(i-1)%n] = - (6. + 3. * float(epsilon) * (X[i][0] + 2*X[(i-1)%n]) + 2. * float(epsilon) ) / (12 * DELTA_x)
        H[i][i] = (epsilon * (X[(i+1)%n][0] - X[(i-1)%n][0])) / (4 *DELTA_x)
        H[i][(i+1)%n] = (6. + 3. * float(epsilon) * (X[i][0] + 2*X[(i+1)%n]) - 2. * float(epsilon) ) / (12 * DELTA_x)
        H[i][(i+2)%n] = float(epsilon) / (12 * DELTA_x)
    return H

def Equ_wave (previous_U):
    """Function returning lambda function whose roots must be found to solve the wave equation (returning the state of the wave following previous_U)"""
    return lambda U: (U-previous_U)/DELTA_t+G((U + previous_U)/2)

def H_wave (previous_U):
    """Functions that calculates the jacobian matrix of the function returned by Equ_wave """
    H1 = np.dot(np.eye(n,n), 1./DELTA_t)
    H2 = np.dot(np.eye(n,n), 2)
    return lambda X: H1+np.dot(H_G(np.dot((X+previous_U),1./2)),H2)

def main():
    letsdisplay = plt.figure()
    plan = plt.axes(xlim=(-L,L), ylim=(-0.4,1.2))
    curve = plan.plot([], [])
    #U = ComputeU0()
    #Draw(ComputeU0())
    
    U = ComputeU0()
    print("Calcul du 30eme etat :")
    for j in range (30):
        U=Newton_BT(Equ_wave(U), H_wave(U), U, 20, 0.0001)
    Draw(U)

main()
        
    

