# EP 1 - Machine Learning MNIST - Poli USP

Versão: Python

## 1. Instruções para compilação e execução do programa:

Vale notar que quando se usou 'g' nos códigos, estava se referindo a variavél que armazenava a quantidade de parâmetros (p no enunciado).

* TAREFA 1

Os arquivos para essa tarefa estão na pasta "Primeira tarefa".

Programa para tarefas a) e b):

Devem ser definidos os números de linhas e de colunas da matriz W a ser resolvida. A matriz b possuirá
dimensões nx1. O programa devolve as matrizes W e b, o parâmetro "x", que são as soluções para as
matrizes W e b, e devolve também a matriz W.x, oriunda da multiplicação das matrizes W e x, 
utilizada para checar quão próxima a solução
obtida está da real.

Programa para tarefas c) e d):

Devem ser definidos as dimensões da matriz W e o número de sistemas simultâneos a serem resolvidos, 
ou seja, o número de colunas da solução e da matriz b. Assim como nas tarefas a) e b), O programa 
devolve as matrizes W e b, a matriz x, contendo as soluções para os sistemas simultâneos e a matriz 
W.x, oriunda da multiplicação das matrizes W e x, utilizada para checar quão próxima a solução obtida
está da real.

* TAREFA 2

Os arquivos para essa tarefa estão na pasta "Segunda tarefa".

Para o programa da tarefa 2, deve ser fornecida a matriz A a ser fatorada. O programa utiliza o
método da fatoração por matrizes não negativas e retorna duas matrizes: W e H. O programa retorna
também a matriz WH, que serve para checar a precisão da fatoração, visto que WH deve ser igual ou
parecida com A.

Exemplo para definição da variável:

"
A = np.array([[3/10,3/5,0],[1/2,0,1],[4/10,4/5,0]])

"

* TAREFA PRINCIPAL

Os arquivos para essa tarefa estão na pasta "Tarefa Principal".

A execução da tarefa principal se baseia em dois códigos. Para que seja possível a realização do treino e do teste, é necessário que os arquivos da base de dados estejam na mesma pasta que o .py (não colocamos por conta do limite de upload do Graúna). Então, adicione-os antes de executar os códigos.

O código treino.py cuida da parte de treinar os classificadores, tendo como entradas (por meio do da função input) a quantidade de dígitos para o treino (ndig_treino) e quantidade de parâmetros, retornando os classificadores em arquivos .txt, cuja nomenclatura tem o seguinte significado:
W[i]nt[ndig_treino]p[p].txt
Onde i é o dígito correspondente ao classficador.

Já o código teste.py realiza os testes do classificadores para verificar sua acurácia. Suas entradas são ndig_treino e p, para que o programa encontre os arquivos .txt criados no treino (eles devem ter sido criados já), e ndig_teste, que traz a quantidade de dígitos testados. O programa retorna, por meio de prints os resultados de taxa de acerto, ocorrencias e acertos para cada dígito, além da taxa de acertos geral, e também um .txt de nome "resultadosnt[ndig_treino]p[p].txt", que armazena as mesmas informações.

Há uma pasta chamada "Exemplo de retorno" que traz os .txt dos classificadores e os resultados do teste para ndig_treino = 100 e p = 5.
