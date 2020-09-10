# EP 1 - Machine Learning MNIST - Poli USP

Vers�o: Python

## 1. Instru��es para compila��o e execu��o do programa:

Vale notar que quando se usou 'g' nos c�digos, estava se referindo a variav�l que armazenava a quantidade de par�metros (p no enunciado).

* TAREFA 1

Os arquivos para essa tarefa est�o na pasta "Primeira tarefa".

Programa para tarefas a) e b):

Devem ser definidos os n�meros de linhas e de colunas da matriz W a ser resolvida. A matriz b possuir�
dimens�es nx1. O programa devolve as matrizes W e b, o par�metro "x", que s�o as solu��es para as
matrizes W e b, e devolve tamb�m a matriz W.x, oriunda da multiplica��o das matrizes W e x, 
utilizada para checar qu�o pr�xima a solu��o
obtida est� da real.

Programa para tarefas c) e d):

Devem ser definidos as dimens�es da matriz W e o n�mero de sistemas simult�neos a serem resolvidos, 
ou seja, o n�mero de colunas da solu��o e da matriz b. Assim como nas tarefas a) e b), O programa 
devolve as matrizes W e b, a matriz x, contendo as solu��es para os sistemas simult�neos e a matriz 
W.x, oriunda da multiplica��o das matrizes W e x, utilizada para checar qu�o pr�xima a solu��o obtida
est� da real.

* TAREFA 2

Os arquivos para essa tarefa est�o na pasta "Segunda tarefa".

Para o programa da tarefa 2, deve ser fornecida a matriz A a ser fatorada. O programa utiliza o
m�todo da fatora��o por matrizes n�o negativas e retorna duas matrizes: W e H. O programa retorna
tamb�m a matriz WH, que serve para checar a precis�o da fatora��o, visto que WH deve ser igual ou
parecida com A.

Exemplo para defini��o da vari�vel:

"
A = np.array([[3/10,3/5,0],[1/2,0,1],[4/10,4/5,0]])

"

* TAREFA PRINCIPAL

Os arquivos para essa tarefa est�o na pasta "Tarefa Principal".

A execu��o da tarefa principal se baseia em dois c�digos. Para que seja poss�vel a realiza��o do treino e do teste, � necess�rio que os arquivos da base de dados estejam na mesma pasta que o .py (n�o colocamos por conta do limite de upload do Gra�na). Ent�o, adicione-os antes de executar os c�digos.

O c�digo treino.py cuida da parte de treinar os classificadores, tendo como entradas (por meio do da fun��o input) a quantidade de d�gitos para o treino (ndig_treino) e quantidade de par�metros, retornando os classificadores em arquivos .txt, cuja nomenclatura tem o seguinte significado:
W[i]nt[ndig_treino]p[p].txt
Onde i � o d�gito correspondente ao classficador.

J� o c�digo teste.py realiza os testes do classificadores para verificar sua acur�cia. Suas entradas s�o ndig_treino e p, para que o programa encontre os arquivos .txt criados no treino (eles devem ter sido criados j�), e ndig_teste, que traz a quantidade de d�gitos testados. O programa retorna, por meio de prints os resultados de taxa de acerto, ocorrencias e acertos para cada d�gito, al�m da taxa de acertos geral, e tamb�m um .txt de nome "resultadosnt[ndig_treino]p[p].txt", que armazena as mesmas informa��es.

H� uma pasta chamada "Exemplo de retorno" que traz os .txt dos classificadores e os resultados do teste para ndig_treino = 100 e p = 5.
