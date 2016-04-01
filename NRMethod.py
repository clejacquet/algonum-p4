# coding: utf-8
import numpy as np


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


def main():
    functionU=Newton_Raphson(lambda A: np.array([A[0]*A[0]-2,A[1]*A[1]-3]),lambda A:  np.array([[(2*A[0]), 0], [0, (2*A[1])]]),np.array([1, 1]),100,0.0001)
    
    print("TEST 1, USING f:(x,y)->(x²-2,y²-3)")
    print("Expected solution :")
    print("[~1.41421 ~1.73205]")
    print("")
    print("Solution we get :")
    print(functionU)

main()
