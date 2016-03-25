# coding: utf-8
import numpy as np

## Newton-Raphson Method

def Newton_Raphson(f,H,U0,N,epsilon):
    U=U0
    UplusV=U-(f(U)/H(U))
    i=1
    while(i<N and ((np.linalg.norm(UplusV-U))>epsilon)):
        U=UplusV
        UplusV=U-(f(U)/H(U))
        i+=1
    return UplusV

def main():
    functionU=Newton_Raphson((lambda (x,y): (x*x-2,y*y-3)),(lambda (x,y): np.mat('[2*x 0 ; 0 2*y]')),(1,1),100,0.0001)
    print("ok")

main()
    
    
