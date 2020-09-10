''' 
Código para o treinamento dos classificadores

Entradas: ndig_treino, p
Saída: Arquivos .txt com os classificador para cada dígito

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
            r =  random.randrange(0,100,1) #Gera um numero aleatorio entre 0 e 100, podendo estar espaçados de 1
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

def MMQA(A,n,m,g):
    ead = 0.00001 #diferenca maxima ente os erros de duas iterações do MMQA 
    itmax = 100 # número maximo de iterações admissível
    W = setting(n,g) #chama a funcao que cria a matriz randomica
    t=1 #variavel que armazena o numero da iteracao
    W = normalize(W,n,g) #normaliza a matriz aleatoria gerada
    An = np.array(A) # Cria copia de A
    At = np.transpose(An) #Armazena a transposta de A
    while t <= itmax: # Enquanto a quantidade de iteracoes for menos que o maximo estipulado
        print("iteração:"+str(t))
        h = solver(np.array(W),np.array(A),n,m,g) # Encontra h para W e A (resolve multiplos sistemas)
        # É utilizado deepcopy nessas funcoes pelo fato de alterarem as matrizes originais durante sua execucao
        for i in range(0,g): # Essa secao igual os elementos de h menores que 0 a 0
            for j in range(0,m):
                if h[i,j] < 0:
                        h[i,j] = 0
        ht = np.transpose(h) # transpoe h
        Wt = solver(np.array(ht),np.array(At),m,n,g) # Encontra Wt para ht e At
        W = np.transpose(Wt) # Encontra W a partir da transposta de Wt
        for i in range(0,n): # Zera os valores negativos de W
            for j in range(0,g):
                if W[i,j] < 0: 
                    W[i,j] = 0
        W = normalize(W,n,g) # Normaliza W
        en = error(np.array(W),np.array(A),np.array(h),n,m,g) # Calcula o erro 
        if t == 1: # Na primeira execucao, igual o erro antigo ao erro novo
            eant = en
            t+=1 # armazena a informacao de que se passou uma iteracao
        elif t > 1: # nas outras iteracoes, verifica se a diferenca enter os erros esta na precisao esperada
            if abs((en-eant)/en) < ead: # se estiver, termina a iteracao
                t = itmax+1 
            else: # se nao estiver, igual o erro antigo ao erro novo e itera novamente
                eant = en
                t+=1
    return W

def treino(ndig_treino,p): #Funcao que treina os Wd
    W=[] # Cria a matriz que irá armazenar os W para cada dígito
    n=784 # Define a dimensao n da matriz, sempre e784 por conta do tamanho das imagens utilizadas
    for i in range(0,10): # Para os 10 digitos
        print("Treinando digito: "+str(i))
        file = "train_dig"+str(i)+".txt" # Cria a string do nome do arquivo
        aux = np.loadtxt(file) # Lê o arquivo
        aux1 = aux[:,:ndig_treino]/255 # Seleciona a quantidade de numeros que se deseja utilizar para treinar e ja divide a matriz por 255
        W.append(MMQA(aux1,n,ndig_treino,p)) # Armazena o Wd na matriz W
    return W

ndig_treino = int(input("Insira a quantidade de dígitos para o treino:")) #Rotina para entrada dos parâmetros de treino e teste
p = int(input("Insira a quantidade de parâmetros:"))
W = treino(ndig_treino,p) # Função que gera os classificadores
for i in range(0,10): # Armazena os classificadores de cada dígito para a condição gerada
    np.savetxt('W'+str(i)+'nt'+str(ndig_treino)+'p'+str(p)+'.txt', W[i], delimiter = ' ')
