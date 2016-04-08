# coding: utf-8
import numpy as np
import math as m
import matplotlib.pyplot as plt
import time as tm

# Newton-Raphson Method
def Newton_Raphson(f, H, U0, N, epsilon):
    """Newton_Raphson takes 5 arguments :
    f: studied function (given as an array)
    H: jacobian matrix of f
    U0: start point
    N: maximum of iterations
    epsilon: precision
    and returns the calculated zeros of f """
    U=U0
    V=np.linalg.lstsq(H(U),-f(U))[0] #function resolving Ax=B
    UplusV=U+V
    i=1
    while(i<N and ((np.linalg.norm(UplusV-U))>epsilon)):
        U=UplusV
        V=np.linalg.lstsq(H(U),-f(U))[0]
        UplusV=U+V
        i+=1
    return UplusV

# Newton Method with Backtracking
def Newton_BT(f,H,U0,N,epsilon):
    """Newton_BT takes 5 arguments :
    f: studied function (given as an array)
    H: jacobian matrix of f
    U0: start point
    N: maximum of iterations
    epsilon: precision
    and returns the zeros of f,
    calculated through the Newton method with backtracking"""
    U=U0
    for i in range(1,N):
        Va=f(U)
        Na=np.linalg.norm(Va)
        if (Na<epsilon):
            return U
        dV=H(U)
        dU=-1*np.linalg.lstsq(dV,Va)[0]
        lambd=1.0
        while (np.linalg.norm(f(U+lambd*dU))>=Na):
            lambd=lambd*(2.0/3)
        U=U+lambd*dU
    return U


def error_NR_depending_on_N(f,H,U0,N1,N2,step,real_zero,eps):
    """error_NR_depending_on_N prints the curve
    representing Error on the calculated zeros depending on the max of iterations of
    the NRMethod"""

    y=[]
    nb_point=int(float((N2-N1))/float(step))+1
    x=np.linspace(N1,N2,nb_point)

    Nzero=np.linalg.norm(real_zero)#exact zero norm

    while(N1<=N2):
        zero=Newton_Raphson(f,H,U0,N1,eps)
        y.append(abs(Nzero-np.linalg.norm(zero)))        
        N1+=step
    plt.plot(x,y)
    plt.ylabel("Error from the NR Method")
    plt.xlabel("Max of iterations of the NR Method")
    plt.title("Error on the calculated zeros depending on the max of iterations of the NRMethod")
    plt.show()


def main():
    functionU2D=Newton_Raphson(lambda A: np.array([A[0]*A[0]-2,A[1]*A[1]-3]),lambda A:  np.array([[(2*A[0]), 0], [0, (2*A[1])]]),np.array([1, 1]),100,0.00000001)

    print("TEST 1 NR, USING f:(x,y)->(x²-2,y²-3)")
    print("Expected solution :")
    print("[~1.41421 ~1.73205]")
    print("Solution we get :")
    print(functionU2D)
    print("")
    print("")
    
    functionU1D=Newton_Raphson(lambda A: np.array([(A[0]**2)-2]),lambda A:  np.array([[2*A[0]]]),np.array([1]),100,0.00000001)

    print("TEST 2 NR, USING f:(x)->(x²-2)")
    print("Expected solution :")
    print("[~1.41421]")
    print("Solution we get :")
    print(functionU1D)
    print("")
    print("")

    functionUBT=Newton_BT(lambda A: np.array([A[0]*A[0]-2,A[1]*A[1]-A[1]-1]),lambda A:  np.array([[(2*A[0]), 0], [0, (2*A[1])-1]]),np.array([2,2]),100,0.000000000001)

    print("TEST 1 BT, USING f:(x,y)->(x²-2,y²-y-1)")
    print("Expected solution :")
    print("[~1.41421 ~1.61803]")
    print("Solution we get :")
    print(functionUBT)
    print("")
    print("")

    functionUBT2=Newton_BT(lambda A: np.array([(A[0]**4)-2*(A[0]**3)-(A[0]**2)+2*A[0]+0.3]),lambda A:  np.array([[4*(A[0]**3)-6*(A[0]**2)-2*A[0]+2]]),np.array([-0.55]),100,0.001) 
    
    print("TEST 2 BT, USING f:(x,y)->X⁴-2X³-X²+2X+0.3")
    print("Expected solution :")
    print("[~ -0.142915]")
    print("Solution we get :")
    print(functionUBT2)
    print("")
    print("")
    
    error_NR_depending_on_N(lambda A: np.array([(A[0]**2)-9]),lambda A:  np.array([[2*A[0]]]),np.array([1]),0,8,1,np.array([3]),0.000001)

main()

