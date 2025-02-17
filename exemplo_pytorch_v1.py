# -*- coding: utf-8 -*-
"""exemplo_pytorch_v1.ipynb

## Tutorial Introdutório de Pytorch (v1)
Traduzido e adaptado do site oficial do python: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html por Hemerson Pistori (pistori@ucdb.br)

## Carregando um banco de imagens
"""

import torch   # Biblioteca pytorch principal
from torch import nn  # Módulo para redes neurais (neural networks)
from torch.utils.data import DataLoader # Manipulação de bancos de imagens
from torchvision import datasets # Ajuda a importar alguns bancos já prontos e famosos
from torchvision.transforms import ToTensor # Realiza transformações nas imagens
import matplotlib.pyplot as plt # Mostra imagens e gráficos

# Definindo alguns hiperparâmetros importantes:
epocas = 10  # Total de passagens durante a aprendizagem pelo conjunto de imagens
tamanho_lote = 64  # Tamanho de cada lote sobre o qual é calculado o gradiente
taxa_aprendizagem = 0.001   # Magnitude das alterações nos pesos

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

# Cria os objetos que irão manipular os dados
train_dataloader = DataLoader(training_data, batch_size=tamanho_lote)
val_dataloader = DataLoader(val_data, batch_size=tamanho_lote)

# Mostra informações do primeiro lote de imagens 
# X vai conter um lote de imagens
# y vai conter as classes (tipo de roupa) de cada imagem do lote
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
     1: "Calças",
     2: "pulôver",
     3: "Vestido",
     4: "Casaco",
     5: "Sandália",
     6: "Camisa",
     7: "Tênis",
     8: "Bolsa",
     9: "Bota de Tornozelo",    
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

        # Cria uma sequência com 3 camadas
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
        # última camada (é aqui que acontece a ativação dos neurônios)
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


# Define a função para treinar a rede
# dataloader = módulo que manipula o conjunto de imagens
# model = arquitetura da rede
# loss_fn = função de perda
# optimizer = otimizador 
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    # Pega um lote de imagens de cada vez do conjunto de treinamento
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)  # Prepara os dados para o dispositivo (GPU ou CPU)
        pred = model(X)         # Realiza uma previsão usando os pesos atuais
        loss = loss_fn(pred, y) # Calcula o erro com os pesos atuais

        optimizer.zero_grad()  # Zera os gradientes pois vai acumular para todas
                               # as imagens do lote
        loss.backward()        # Retropropaga o gradiente do erro
        optimizer.step()       # e recalcula todos os pesos da rede

        # Imprime informação a cada 100 lotes processados 
        if batch % 100 == 0:
            # Mostra a perda e o total de imagens já processadas
            loss, current = loss.item(), batch * len(X)
            print(f"Perda: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define a função de validação
def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # Total de imagens para validação
    num_batches = len(dataloader)   # Total de lotes
    model.eval()  # Coloca a rede em modo de avaliação (e não de aprendizagem)
    # Vai calcular o erro no conjunto de validação
    val_loss, correct = 0, 0

    # Na validação os pesos não são ajustados e por isso não precisa
    # calcular o gradiente
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    acuracia = correct / size
    print("Informações na Validação:")
    print(f"Total de acertos: {int(correct)}")
    print(f"Total de imagens: {size}")
    print(f"Perda média: {val_loss:>8f}")            
    print(f"Acurácia: {(100*acuracia):>0.1f}%")

"""## Treinando a Rede Neural (Aprendizagem)"""

# Passa por todas as imagens várias vezes (a quantidade de vezes
# é definida pelo hiperparâmetro "epocas")

for t in range(epocas):
    print(f"-------------------------------")
    print(f"Época {t+1}\n-------------------------------")
    train(train_dataloader, model, funcao_perda, otimizador)
    validation(val_dataloader, model, funcao_perda)

print("Terminou a fase de aprendizagem !")

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

"""## Usando a rede treinada para classificar imagens"""

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

# Vai mostrar a classificação da rede para 9 imagens escolhidas
# aleatoriamente do conjunto de validação
figure = plt.figure(figsize=(8, 8))  # Cria o local para mostrar as imagens
cols, rows = 3, 3  # Irá mostrar 9 imagens em uma grade 3x3
for i in range(1, cols * rows + 1):
    # Gera um número aleatório menor que o total de imagens disponíveis
    sample_idx = torch.randint(len(val_data), size=(1,)).item()
    # Pega a imagem e sua classe usando o número aleatório
    img, label = val_data[sample_idx]
    # Classifica a imagem usando a rede treinada
    predita = classifica_uma_imagem(model,img,label)
    # Adiciona a imagem na grade que será mostrada
    figure.add_subplot(rows, cols, i)
    # Usa a classe da imagem como título da imagem
    plt.title(predita)
    # Não mostra valores para os eixos X e Y
    plt.axis("off")
    # Avisa que é uma imagem em tons de cinza 
    # o squeeze garante que vai pegar apenas um canal da imagem
    plt.imshow(img.squeeze(), cmap="gray")
    
plt.show() # Este é o comando que vai mostrar as imagens
