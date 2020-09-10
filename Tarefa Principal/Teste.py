'''
Código para o teste dos classificadores a patir dos arquivos .txt

Entradas: ndig_treino, p, ndig_teste, Classificadores em .txt

Autores:
    Gustavo Lopes NUSP: 10335490
    Lucas Okamura NUSP: 9274315
'''
import numpy as np #Modulo necessario para utilizar funções basicas para armazenamento de matrizes, criacao de matrizes de zeros, matrizes identidades, operações, como multiplicação de matrizes
import math #modulo que traz operacoes matematicas de certa complexidade para o python
import random #modulo utilizado para gerar matrizes aleatorias

def setting(n,g): #funcao que cria a matriz W randomicamente para iniciar o MMQA
    W = np.zeros((n,g)) #Cria uma matriz de zeros do tamanho de W
    for i in range(0,n): #Percorre a matriz
        for j in range(0,g):
            r =  random.randrange(0,10000,1) #Gera um numero aleatorio entre 0 e 100, podendo estar espaçados de 1
            W[i,j] = r     
    return W

def sencos(W,i,j,k): #Funcao que calcula sen e cos para rotacao de givens
    c = W[i,k]/math.sqrt(W[i,k]*W[i,k]+W[j,k]*W[j,k])
    s = -W[j,k]/math.sqrt(W[i,k]*W[i,k]+W[j,k]*W[j,k])
    return c, s

def givensrot(W,b,i,j,c,s): #Funcao que aplica rotacao de givens para W e b
    Wi = W[i,:] # Em vez de utiliar o algoritmo exato apresentado no enunciado
    Wj = W[j,:] # preferiu-se fazer a multiplicação das linhas e colunas de uma
    aux = Wi*c-s*Wj # vez só, para economizar em tempo de execução do programa
    W[j,:] = Wi*s+c*Wj
    W[i,:] = aux
    bi = b[:,i]
    bj = b[:,j]
    aux = bi*c-s*bj
    b[:,j] = bi*s+bj*c
    b[:,i] = aux
    return W, b

def solver(K,A,n,m,g): #Função que resolve varios sistemas simultâneos
    b = np.transpose(A) #Usa-se a transposta de A para ficar mais fácil escolher uma coluna separadamente
    for k in range(0,g): #Essa seção se baseia na aplicação do algoritmo apresentado no enunciado
        for j in range(n-1,k,-1):
            i=j-1
            if K[j,k] != 0: # Quando o elementos analisado não é nulo
                c,s = sencos(K,i,j,k) # Calcula o sen e o cos para rotação de Givens
                K, b = givensrot(K,b,i,j,c,s) # Aplica a rotação de Givens às matrizes W e b
    A = np.transpose(b) #transpoe-se novamente b
    x = solve(K,A,n,m,g) #chama a funcao que resolve o sistema
    return x
      
def solve(W,A,n,m,g): #funcao que resolve o sistema 
    x = np.zeros((g,m)) #cria matriz para acondicionar os resultados
    for k in range(g-1,-1,-1): #Aplica o algoritmo apresentado no enunciado
        for j in range(0,m):
            aux = 0
            for i in range(k,g):
                aux += W[k,i]*x[i,j] #Armazena a soma 
            x[k,j] = (A[k,j] - aux)/W[k,k]
    return x

def error(K,A,x,n,m,g): #funcao que calcula o erro quadrado do MMQA
    soma = 0
    r = np.matmul(K,x) #multiplicam-se as matrizes W e x
    e = np.zeros((n,g)) #cria-se uma matriz para armazenar o erro
    for i in range(n):
        for j in range(g):
            soma += (A[i,j] - r[i,j])**2 #calcula o erro quadrado
    return soma

def normalize(A,n,g): #Funcao que normaliza A
    sj = np.zeros(g) #cria uma matriz para armazenar a soma quadrada dos elementos de cada coluna
    B = np.zeros((n,g)) #Cria uma matriz para armazenar a matriz normalizada
    for j in range(0,g): 
        sij = 0
        for i in range(0,n):
            sij += A[i,j]*A[i,j] #soma o elemento ao quadrado
        sj[j] = np.sqrt(sij)
    for i in range(0,n):
        for j in range(0,g):
            B[i,j] = A[i,j]/sj[j] #faz a divisão de cada elemento da coluna pela soma quadrada dos seus elementos e adiciona na matriz normalizada
    
    return B

def test(ndig_test,W,p): # Funcao que testa os classificadores criados
    n = 784
    file = "test_images.txt" 
    test = np.loadtxt(file) #Carrega a matriz com os digitos a serem testados
    test = test[:,:ndig_test]/255 # Seleciona a quantidade de digitos a serem testados e ja divide a matriz por 255
    file = "test_index.txt"
    index = np.loadtxt(file) #Carrega a matriz com os digitos verdadeiros de cada numero testado 
    Ct = [] # matrizes que armazenarao os erros
    for k in range(0,10): # para cada digito
        H = solver(np.array(W[k]),np.array(test),n,ndig_test,p) # resolve os multiplos sistemas
        R = test - np.matmul(W[k],H) # verifica o erro entre a matriz A e o produto WH
        c=[]
        for j in range(0,ndig_test):
            cj = np.sqrt(np.sum(R[:,j]**2))
            c.append(cj)
        Ct.append(c) # Armazena o erro de cada classificação
    result = []
    for j in range(0,ndig_test): # Essa secao procura a classificacao com menor erro, que teoricamente classifica corretamente o numero
        for i in range(0,10): # primeiro escolhe um dos numeros testados e depois ve qual classificador de digito o classificou com menor erro
            if i == 0:
                k = 0
                cmin = Ct[i][j]
            if Ct[i][j] < cmin: # Se o erro encontrado na classificação for menor que o menor erro encontrado até o momento
                k = i # Indica que o dígito é o melhor classificador até então
                cmin = Ct[i][j]
        result.append(k) # Matriz de resultados armazena o digito que obteve menor erro
    acertos = 0 # Essa secao consiste em verificar a acuracia dos classificadores
    digito = np.zeros(10) # Armazena quantas vezes o digito aparece
    digitocerto = np.zeros(10) # Armazena quantas vezes cada digito foi classificado corretamente
    for i in range(0,ndig_test): # Para cada um numero testado
        digito[int(index[i])] += 1 # Adiciona um a contagem de quantas vezes o digito apareceu (utiliza o indice de digitos conhecidos)
        if result[i] == index[i]: # Se classificou-se corretamente
            acertos +=1 # Conta-se mais um na quantidade de acertos geral
            digitocerto[int(index[i])] += 1 # Conta-se mais um na quantidade de acertos para aquele digito
    percentualtotal = acertos/ndig_test # Calcula a porcentagem total de acertos
    percentual = []
    for i in range(0,10): #Calcula a porcentagem de acertos para cada digito
        percentual.append(digitocerto[i]/digito[i])
    return result, percentualtotal, percentual, digito, digitocerto
ndig_treino = int(input("Quantidade de dígitos usada para o treino do classificador:")) #Rotina para entrada dos parâmetros de treino e teste
p = int(input("Insira a quantidade de parâmetros usada no classificador:"))
ndig_test = int(input("Insira a quantidade de dígitos para o teste:"))
W = []
for i in range(0,10):
    W.append(np.loadtxt('W'+str(i)+'nt'+str(ndig_treino)+'p'+str(p)+'.txt'))
result, percentualtotal, percentual, digito, digitocerto = test(ndig_test,W,p)
print("Índice de acertos total:") # Printa o índice total de acertos
print(percentualtotal) 
for i in range(0,10): # printa o índice de acertos, a quantidade de ocorrências e quantidade de acertos para cada dígito
    print("Índice de acertos para o dígito "+str(i)+":")
    print(percentual[i])
    print("Número de ocorrências desse dígito na base de dados de teste:")
    print(digito[i])
    print("Número de acertos para esse dígito:")
    print(digitocerto[i])
texto = [] # Essa seção do código trata de gerar um arquivo com os resultados de eficácia dos classificadores
legenda = "Digito \ Taxa de acertos \ No de ocorrências \ No de acertos"
texto.append(legenda)
for i in range(0,10):
    dig = str(i)+' \ '+str(round((percentual[i]),4))+' \ '+str(int(digito[i]))+' \ '+str(int(digitocerto[i]))
    texto.append(dig)
geral = 'Taxa de acertos geral: '+str(round(percentualtotal,4))
texto.append(geral)
np.savetxt('resultadosnt'+str(ndig_treino)+'p'+str(p)+'.txt', texto, delimiter = ' ', fmt="%s")