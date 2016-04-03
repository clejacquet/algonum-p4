# coding: utf-8
import numpy as np
import time as tm

# Newton-Raphson Method
def Newton_Raphson(f, H, U0, N, epsilon):
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


def Newton_BT(f,H,U0,N,epsilon):
    U=U0
    for i in range(1,N):
        print(i)
        Va=f(U)
        Na=np.linalg.norm(Va)
        if (Na<epsilon):
            print("Norme < epsilon")
            return U
        dV=H(U)
        dU=-1*np.linalg.lstsq(dV,Va)[0]
        lambd=1.0
        print("norme: ",(np.linalg.norm(f(U+lambd*dU))))
        print("Na: ",Na)
        while (np.linalg.norm(f(U+lambd*dU))>=Na):
            lambd=lambd*(2.0/3)
            print("LAMBDA: ",lambd)
        U=U+lambd*dU
    return U


def main():
    # functionU=Newton_Raphson(lambda A: np.array([A[0]*A[0]-2,A[1]*A[1]-3]),lambda A:  np.array([[(2*A[0]), 0], [0, (2*A[1])]]),np.array([1, 1]),100,0.00000001)

    
    # print("TEST 1, USING f:(x,y)->(x²-2,y²-3)")
    # print("Expected solution :")
    # print("[~1.41421 ~1.73205]")
    # print("")
    # print("Solution we get :")
    # print(functionU)

    # functionUBT=Newton_BT(lambda A: np.array([A[0]*A[0]-2,A[1]*A[1]-A[1]-1]),lambda A:  np.array([[(2*A[0]), 0], [0, (2*A[1])-1]]),np.array([2,2]),100,0.000000000001)

    # print("TEST 1 BT, USING f:(x,y)->(x²-2,y²-y-1)")
    # print("Expected solution :")
    # print("[~ ~]")
    # print("")
    # print("Solution we get :")
    # print(functionUBT)

    functionUBT2=Newton_BT(lambda A: np.array([(A[0]**4)-2*(A[0]**3)-(A[0]**2)+2*A[0]+0.3]),lambda A:  np.array([[4*(A[0]**3)-6*(A[0]**2)-2*A[0]+2]]),np.array([-0.55]),100,0.001)

    print("TEST 2 BT, USING f:(x,y)->X⁴-2X³-X²+2X+0.3")
    print("Expected solution :")
    print("[]")
    print("")
    print("Solution we get :")
    print(functionUBT2)

    
## Faire tracé de l'erreur
## Faire test 1D de NR
## Faire Nexton Backtracking



main()

