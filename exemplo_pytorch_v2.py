# -*- coding: utf-8 -*-
"""
## Tutorial Introdutório de Pytorch (v2)
Traduzido e adaptado do site oficial do python: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html por Hemerson Pistori (pistori@ucdb.br)

Mudanças desta versão:

- Implementação de parada antecipada (Early Stopping)
- Leitura de uma conjunto de teste a partir das imagens em pastas
- Mostrar gráfico de perda e acurácia por época
- Mostrar métricas de precisão, revocação e medida-F por classe e total
- Mostrar matriz de confusão

## Carregando um banco de imagens
"""

import torch   # Biblioteca pytorch principal
from torch import nn  # Módulo para redes neurais (neural networks)
from torch.utils.data import DataLoader # Manipulação de bancos de imagens
from torchvision import datasets # Ajuda a importar alguns bancos já prontos e famosos
from torchvision.transforms import ToTensor # Realiza transformações nas imagens
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # Mostra imagens e gráficos
from torch.utils.tensorboard import SummaryWriter # Salva "log" da aprendizagem
import torchvision
import PIL  # Biblioteca para manipulação de imagens
import sklearn.metrics as metrics  # Ajuda a calcular métricas de desempenho
from sklearn.metrics import precision_recall_fscore_support as score
import seaborn as sn  # Usado para gerar um mapa de calor para a matriz de confusão
import pandas as pd   # Ajuda a trabalhar com tabelas
import numpy as np    # Várias funções numéricas

# Definindo alguns hiperparâmetros importantes:
epocas = 50  # Total de passagens durante a aprendizagem pelo conjunto de imagens
tamanho_lote = 64  # Tamanho de cada lote sobre o qual é calculado o gradiente
taxa_aprendizagem = 0.001   # Magnitude das alterações nos pesos
paciencia = 5  # Total de épocas sem melhoria da acurácia na validação até parar
tolerancia = 0.01 # Melhoria menor que este valor não é considerada melhoria

# As imagens de teste, que eu peguei da Internet e não estão nem no conjunto
# de treinamento e nem de validação, ficarão nesta pasta:
pasta_imagens_teste = "data/FashionMNIST_custom_testset/"

# Definindo os dados para treinamento da rede neural
# Utiliza uma base de imagens de roupas chamada FashionMNIST
training_data = datasets.FashionMNIST(
    root="data",  # Pasta onde ficarão os dados
    train=True,   # Usa apenas dados de treinamento
    download=True,  # Faz download dos dados pela Internet 
    transform=ToTensor(),  # Converte do formato jpg para um tensor
)

# Definindo os dados para validação da rede neural
val_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Cria os objetos que irão manipular os dados (basicamente ajuda a pegar
# lote (batch) de imagens de treinamento e de validação)
train_dataloader = DataLoader(training_data, batch_size=tamanho_lote)
val_dataloader = DataLoader(val_data, batch_size=tamanho_lote)

# Mostra informações do primeiro lote de imagens 
# X vai conter um lote de imagens
# y vai conter as classes (tipo de vestimenta) de cada imagem do lote
for X, y in val_dataloader:
    print(f"Tamanho do lote de imagens: {X.shape[0]}")
    print(f"Quantidade de canais: {X.shape[1]}")
    print(f"Altura de cada imagem: {X.shape[2]}")
    print(f"Largura de cada imagem: {X.shape[3]}")
    print(f"Tamanho do lote de classes (labels): {y.shape[0]}")
    print(f"Tipo de cada classe: {y.dtype}")
    break  # Para depois de mostrar os dados do primeiro lote

print(f"Total de imagens de treinamento: {len(training_data)}")
print(f"Total de imagens de validação: {len(val_data)}")

"""### Mostrando algumas imagens"""

# Definindo o nome de cada classe (que a principio é apenas
# um número)
labels_map = {
     0: "Camiseta",
     1: "Calcas",
     2: "Pulover",
     3: "Vestido",
     4: "Casaco",
     5: "Sandalia",
     6: "Camisa",
     7: "Tenis",
     8: "Bolsa",
     9: "Bota",    
}

figure = plt.figure(figsize=(8, 8))  # Cria o local para mostrar as imagens
cols, rows = 3, 3  # Irá mostrar 9 imagens em uma grade 3x3
for i in range(1, cols * rows + 1):
    # Gera um número aleatório menor que o total de imagens disponíveis
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # Pega a imagem e sua classificação usando o número aleatório
    img, label = training_data[sample_idx]
    # Adiciona a imagem na grade que será mostrada
    figure.add_subplot(rows, cols, i)
    # Usa a classe da imagem como título da imagem
    plt.title(labels_map[label])
    # Não mostra valores para os eixos X e Y
    plt.axis("off")
    # Avisa que é uma imagem em tons de cinza 
    # o squeeze garante que vai pegar apenas um canal da imagem
    plt.imshow(img.squeeze(), cmap="gray")
    
plt.show() # Este é o comando que vai mostrar as imagens

"""## Definindo uma rede neural artificial"""

# Verifica se tem GPU na máquina, caso contrário, usa a CPU mesmo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device}")

# Define uma rede neural artificial a partir 
# da classe nn do pytorch
class NeuralNetwork(nn.Module):
    def __init__(self):

        # Inicializa a classe "pai"
        super(NeuralNetwork, self).__init__()

        # Cria uma camada para achatar a imagem (transformar
        # de 2 dimensões para uma dimensão)
        self.flatten = nn.Flatten()

        # Cria uma sequência com 3 camadas (ou 5 camadas, se considerar
        # a ativação com Relu como sendo uma camada também)
        #
        # - uma linear com 784 (28*28) neurônios entrando e 512 saindo e ativação ReLU
        # - outra linear com 512 neurônios entrando e 512 saindo e ativação ReLU
        # - e a última com 512 neurônios entrando e 10 saindo (as 10 classes
        #   do problema)
        #
        # A camada linear é tambem chamada de completamente conectada ou Densa
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    # Define como funcionará o passo "para frente" (forward)
    # do algoritmo de retropropagação (backpropagation)
    def forward(self, x):
        # Realiza o achatamento do tensor 
        # Transforma um matriz 28*28 em um vetor com 784 posições
        x = self.flatten(x)
        # Aplica todas as camadas em sequência e guarda o resultado final da
        # última camada 
        output_values = self.linear_relu_stack(x)
        return output_values

# Prepara a rede para o dispositivo que irá processá-la
model = NeuralNetwork().to(device)

# Imprime dados sobre a arquitetura da rede
print(model)

# Define o otimizador como sendo descida de gradiente estocástica
otimizador = torch.optim.SGD(model.parameters(), lr=taxa_aprendizagem)

# Define a função de perda como entropia cruzada
funcao_perda = nn.CrossEntropyLoss()

# Cria o módulo do tensorboard de coleta de dados
writer = SummaryWriter()

# Define a função para treinar a rede
# dataloader = módulo que manipula o conjunto de imagens
# model = arquitetura da rede
# loss_fn = função de perda
# optimizer = otimizador 
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # Total de imagens
    num_batches = len(dataloader)   # Total de lotes
    model.train()  # Avisa que a rede vai entrar em modo de aprendizagem

    train_loss, train_correct = 0, 0  # Usado para calcular perda e acurácia médias

    # Pega um lote de imagens de cada vez do conjunto de treinamento
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)  # Prepara os dados para o dispositivo (GPU ou CPU)
        pred = model(X)         # Realiza uma previsão usando os pesos atuais
        loss = loss_fn(pred, y) # Calcula o erro com os pesos atuais

        train_loss += loss_fn(pred, y).item() # Guarda para calcular a perda média
        # Calcula os acertos para o lote inteiro de imagens
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item() 

        optimizer.zero_grad()  # Zera os gradientes pois vai acumular para todas
                               # as imagens do lote
        loss.backward()        # Calcula os gradientes com base no erro (loss)
        optimizer.step()       # Ajusta os pesos com base nos gradientes

        # Imprime informação a cada 100 lotes processados 
        if batch % 100 == 0:
            # Mostra a perda e o total de imagens já processadas
            loss, current = loss.item(), batch * len(X)
            print(f"Perda: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches  # Como a perda foi calculada por lote, divide
                               # pelo total de lotes para calcular a média
    train_acuracia = train_correct / size  # Já o total de acertos é em relação
                                           # ao total geral de imagens

    return train_loss, train_acuracia        

# Define a função de validação (aqui a rede não está aprendendo, apenas
# usando "aquilo que aprendeu", mas em um conjunto de imagens diferente
# do conjunto usado para aprender)
def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # Total de imagens para validação
    num_batches = len(dataloader)   # Total de lotes
    model.eval()  # Coloca a rede em modo de avaliação (e não de aprendizagem)
    
    # Vai calcular a perda e o total de acertos no conjunto de validação
    val_loss, val_correct = 0, 0

    # Na validação os pesos não são ajustados e por isso não precisa
    # calcular o gradiente
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    val_acuracia = val_correct / size

    print("Informações na Validação:")
    print(f"Total de acertos: {int(val_correct)}")
    print(f"Total de imagens: {size}")
    print(f"Perda média: {val_loss:>8f}")            
    print(f"Acurácia: {(100*val_acuracia)}%")
    return val_loss, val_acuracia

"""## Treinando a Rede Neural (Aprendizagem)"""

# A aprendizagem agora tem parada antecipada (early stopping)

maior_acuracia = 0  # Guarda a melhor acurácia no conjunto de validação
total_sem_melhora = 0  # Guarda quantas épocas passou sem melhoria na acurácia

# Passa por todas as imagens várias vezes (a quantidade de vezes
# é definida pelo hiperparâmetro "epocas")
for epoca in range(epocas):
    print(f"-------------------------------")
    print(f"Época {epoca+1} \n-------------------------------")
    train_loss, train_acuracia = train(train_dataloader, model, funcao_perda, otimizador)
    val_loss, val_acuracia = validation(val_dataloader, model, funcao_perda)

    # Guarda informações para o tensorboard pode criar os gráficos depois
    writer.add_scalars('Loss', {'train':train_loss,'val':val_loss}, epoca)
    writer.add_scalars('Accuracy', {'train':train_acuracia,'val':val_acuracia}, epoca)

    # Soma uma tolerancia no valor da maior acurácia para que melhoras muito
    # pequenas não sejam consideradas
    if val_acuracia > (maior_acuracia+tolerancia): 
      maior_acuracia = val_acuracia
      total_sem_melhora = 0
    else: 
      total_sem_melhora += 1 
      print(f"Sem melhora há {total_sem_melhora} épocas ({100*val_acuracia}% <= {100*(maior_acuracia+tolerancia)}%)")
    if total_sem_melhora > paciencia:
      print(f"Acabou a paciência com {epoca+1} épocas ")
      break

print("Terminou a fase de aprendizagem !")

writer.close()

"""## Visualização usando Tensorboard

"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=runs

"""# Salvando a rede treinada


"""

# Salva em disco os pesos da rede treinada para ser usada
# posteriormente (sem precisar aprender novamente)

torch.save(model.state_dict(), "modelo_treinado.pth")
print("Salvou o modelo treinado em modelo_treinado.pth")

"""## Carregando a rede treinada anteriormente e usando


"""

model = NeuralNetwork()
model.load_state_dict(torch.load("modelo_treinado.pth"))

"""## Usando a rede treinada para classificar algumas imagens 

Usando um conjunto de teste próprio que será lido do disco. As imagens são um pouco diferentes do conjunto usado para treinamento da rede e por isso e desempenho cai bastante. 
"""

# É PRECISO RETIRAR OS COMENTÁRIOS ABAIXO SE FOR RODAR NO COLAB

# Vai baixar o banco de imagens de teste, colocar na pasta data e descompactar
#!curl -L -o FashionMNIST_custom_testset.zip "https://drive.google.com/uc?export=download&id=1kH4JxSPkqSwsrWOQd40xYiAnhziK6_iq"

# !mv Fash*.zip ./data/
# %cd ./data/
#!unzip Fash*.zip
# %cd ..



# Classifica uma única imagem 
# model: rede a ser usada
# x: imagem
# y: classificação real da imagem
# predita: classificação dada pela rede
def classifica_uma_imagem(model,x,y):
    model.eval()
    with torch.no_grad():
       pred = model(x)
       predita, real = labels_map[int(pred[0].argmax(0))], labels_map[y]
       print(f'Predita: "{predita}", Real: "{real}"')
    return(predita)

# Cria uma transformação nova que não está disponível diretamente para
# composição de transformação 
def invert(image: PIL.Image.Image) -> PIL.Image.Image:
    return transforms.functional.invert(image)


# Carrega o banco de imagens de teste aplicando as transformações
# necessárias
test_data = datasets.ImageFolder(root=pasta_imagens_teste,
                                    transform=transforms.Compose(
                                        [  transforms.Resize((28,28)),
                                           transforms.Grayscale(num_output_channels=1),
                                           transforms.Lambda(invert),
                                           transforms.ToTensor(),
                                        ])
                                 ) 

# Vai mostrar a classificação da rede para 16 imagens do conjunto de teste
figure = plt.figure(figsize=(8, 8))  # Cria o local para mostrar as imagens
cols, rows = 4, 4  # Irá mostrar 16 imagens em uma grade 4x4
print(f"Testando em {len(test_data)} imagens. Resultados:")
for i in range(cols*rows):
    img, label = test_data[i]
    # Classifica a imagem usando a rede treinada
    predita = classifica_uma_imagem(model,img,label)
    # Adiciona a imagem na grade que será mostrada
    figure.add_subplot(rows, cols, i+1)
    # Usa a classe da imagem como título da imagem
    plt.title(predita)
    # Não mostra valores para os eixos X e Y
    plt.axis("off")
    # Avisa que é uma imagem em tons de cinza 
    # o squeeze garante que vai pegar apenas um canal da imagem
    plt.imshow(img.squeeze(), cmap="gray")
    
plt.show() # Este é o comando que vai mostrar as imagens

"""## Gera matriz de confusão e algumas métricas de avaliação"""

# Listas para guardar valores preditos e reais
predicoes = []
reais = []

# Vai acumular acertos para calcular acurácia
test_correct=0

model.eval() # Coloca a rede no modo de avaliação (e não de aprendizagem)
with torch.no_grad():   # Avisa que não devem ser calculados gradientes
   for img, label in test_data:   # Para cada imagem do conjunto de teste
      predicao = model(img)       # Faz a predição usando a rede
      predicao = int(predicao[0].argmax(0))  # Pega a classe com maior valor
      predicoes.extend([predicao]) # Guarda predição na lista
      reais.extend([label])        # Guarda valor real na lista
      test_correct += (predicao == label)

# Acurácia no conjunto de teste
test_acuracia = test_correct/len(test_data)

# Constroi a matriz de confusão
matriz = metrics.confusion_matrix(reais,predicoes)

# Pega a lista de classes 
classes=list(labels_map.values())

# Transforma a matriz no formato da biblioteca PANDA
df_matriz = pd.DataFrame(matriz/np.sum(matriz), index = classes,
                     columns = [i for i in classes])

# Gera uma imagem do tipo mapa de calor
plt.figure(figsize = (12,7))
sn.heatmap(df_matriz, annot=True)
plt.savefig('matriz_confusao.png')

print('Métricas de desempenho no conjunto de teste:')
print(metrics.classification_report(reais,predicoes))

precision,recall,fscore,support=score(reais,predicoes,average='macro')
print('-----------------------------------')
print(f'Resumo para as {len(test_data)} imagens de teste:')
print(f"Acertos: {int(test_correct)}")
print(f"Acurácia: {(100*test_acuracia):>0.2f}%")
print(f"Precisão: {100*precision:>0.2f}%")
print(f"Revocação: {100*recall:>0.2f}%")
print(f"Medida-F: {100*fscore:>0.2f}%")
print('-----------------------------------')
