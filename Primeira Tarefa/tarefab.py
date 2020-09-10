
"""
Created on Fri Apr 26 20:16:10 2019

Gustavo Lopes NUSP 10335490
Lucas Okamura  NUSP 9274315
"""
import numpy as np #Modulo necessario para utilizar funções basicas para armazenamento de matrizes, criacao de matrizes de zeros, matrizes identidades, operações, como multiplicação de matrizes
import math #modulo que traz operacoes matematicas de certa complexidade para o python
import random #modulo utilizado para gerar matrizes aleatorias


print("Insira as dimensões da matriz W")
n = int(input("Linhas:"))
m = int(input("Colunas:"))

def matrixbuild(n,p): #matriz utilizada para criar a matriz W, dadas suas dimensões e lei de formacao
    K = np.zeros((n,p))
    for i in range(0,n):
        for j in range(0,p):
            if abs(i-j) <= 4:
                K[i][j] = 1/(i+j+1)
            elif abs(i-j) > 4:
                K[i][j] = 0
    return K

def bbuild(n): #matriz utilizada para criar a matriz b, dadas suas dimensoes e lei de formacao
    b = np.zeros((n))
    for i in range(0,n):
        b[i]=i+1
    return b

W = matrixbuild(n,m)
b = bbuild(n)
print(b)


def sencos(W,i,j,k): #Funcao que calcula sen e cos para rotacao de givens
    c = W[i,k]/math.sqrt(W[i,k]*W[i,k]+W[j,k]*W[j,k])
    s = -W[j,k]/math.sqrt(W[i,k]*W[i,k]+W[j,k]*W[j,k])
    return c, s

def givensrot(W,i,j,m,c,s): #Funcao que aplica rotacao de givens para W e b
    Wi = W[i,:]
    Wj = W[j,:]
    aux = Wi*c-s*Wj
    W[j,:] = Wi*s+c*Wj
    W[i,:] = aux
    bi = b[i]
    bj = b[j]
    aux = bi*c-s*bj
    b[j] = bi*s+bj*c
    b[i] = aux
    return W, b

def solver(K,A,n,m): #Função que resolve varios sistemas simultâneos
    b = np.transpose(A) #Usa-se a transposta de A para ficar mais fácil escolher uma coluna separadamente
    for k in range(0,m): #Essa seção se baseia na aplicação do algoritmo apresentado no enunciado
        for j in range(n-1,k,-1):
            i=j-1
            if K[j,k] != 0:
                c,s = sencos(K,i,j,k) 
                K, b = givensrot(K,i,j,m,c,s)
            A = np.transpose(b) #transpoe-se novamente b
    x = solve(K,b,n,m) #chama a funcao que resolve o sistema
    return x


def solve(W,A,n,m): #funcao que resolve o sistema 
    x = np.zeros(m) #cria matriz para acondicionar os resultados
    for k in range(m-1,-1,-1): #Aplica o algoritmo apresentado no enunciado
        aux = 0
        for j in range(k,m):
            aux += W[k,j]*x[j] #Armazena a soma 
        x[k] = (A[k] - aux)/W[k,k]
    return x

W = matrixbuild(n,m) # Gera as matrizes seguindo a lei de formação do enunciado
b = bbuild(n) 
x = solver(np.array(W),np.array(b),n,m) # Encontra a solução do sistema
C = np.matmul(W,x) #serve para checar se a solucao x é próxima da solucao real

print("W =") 
print(W, "\n")
print("x =")
print(x, "\n")
print("b =")
print(b, "\n")
print("W.x =")
print(C)