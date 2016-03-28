# coding: utf-8
import numpy as np

## Newton-Raphson Method

def Newton_Raphson(f,H,U0,N,epsilon):
    U=U0
    V=np.linalg.lstsq(H(U),-f(U))[0]
    UplusV=U+V
    i=1
    while(i<N and ((np.linalg.norm(UplusV-U))>epsilon)):
        U=UplusV
        V=np.linalg.lstsq(H(U),-f(U))[0]
        UplusV=U+V
        i+=1
    return UplusV

def fn_ex(A):
    return (np.array([A[0]*A[0]-2,A[1]*A[1]-3]))

def H_ex(A):
    return np.array([[(2*A[0]), 0], [0, (2*A[1])]])

def main():
    functionU=Newton_Raphson(fn_ex,H_ex,np.array([1, 1]),100,0.0001)
    print("Expected solution :")
    print("[~1.41421 ~1.73205]")
    print("")
    print("Solution we get :")
    print(functionU)
main()
    
    
