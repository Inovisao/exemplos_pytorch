# -*- coding: utf-8 -*-
"""exemplo_pytorch_v4.ipynb

## Tutorial Introdutório de Pytorch (v4)

Mudanças nesta versão:

- Lendo os dados de uma pasta no google drive, separados entre treino (train) e teste (test). Dentro de cada pasta (train,teste) as imagens devem estar separadas em subpastas, uma para cada classe.
- Separa percentual para validação
- Mostra grafo da rede e lote de imagens no tensorboard


### Carregando um banco de imagens
Usa como exemplo imagens disponíveis na pasta 'data' deste projeto aqui: http://git.inovisao.ucdb.br/inovisao/exemplos_pytorch

Dentro de exemplos_pytorch, na pasta data, tem um script BASH (Linux) chamado split_train_test.sh para separar as suas imagens em treino e teste (dentro de script tem uma explicação de como usá-lo). É preciso usar o terminal Linux para executá-lo. Use chmod 755 ./split*.sh para transformar o script em executável, se necessário.
"""

import torch   # Biblioteca pytorch principal
from torch import nn  # Módulo para redes neurais (neural networks)
from torch.utils.data import DataLoader # Manipulação de bancos de imagens
from torchvision import datasets,models # Ajuda a importar alguns bancos e
                                        # e modelos já prontos e famosos
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # Mostra imagens e gráficos
from torch.utils.tensorboard import SummaryWriter # Salva "log" da aprendizagem
from torch.utils.data import Subset
import torchvision
import PIL  # Biblioteca para manipulação de imagens
import sklearn.metrics as metrics  # Ajuda a calcular métricas de desempenho
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
import seaborn as sn  # Usado para gerar um mapa de calor para a matriz de confusão
import pandas as pd   # Ajuda a trabalhar com tabelas
import numpy as np    # Várias funções numéricas

# Definindo alguns hiperparâmetros importantes:
epocas = 100  # Total de passagens durante a aprendizagem pelo conjunto de imagens
tamanho_lote = 16  # Tamanho de cada lote sobre o qual é calculado o gradiente
taxa_aprendizagem = 0.01   # Magnitude das alterações nos pesos
momento = 0.2  # Mantem informação de pesos anteriores (as mudanças de
               # de peso passam a ser mais suaves)
paciencia = 5  # Total de épocas sem melhoria da acurácia na validação até parar
tolerancia = 0.01 # Melhoria menor que este valor não é considerada melhoria
perc_val = 0.2    # Percentual do treinamento a ser usado para validação

# Define uma arquitetura já conhecida que será usada
# Opções atuais: "resnet", "squeezenet", "densenet"
nome_rede = "resnet"
tamanho_imagens = 224  # Tamanho das imagens para estas arquiteturas

# Descomente o código abaixo se quiser montar e usar o seu próprio google drive
# no lugar das pastas que o colab cria automaticamente 
# from google.colab import drive
# drive.mount('/content/drive')


# Vai baixar o banco de imagens de treino e de teste do exemplo com peixes
!curl -L -o v4_train_test.zip "https://drive.google.com/uc?export=download&id=1aW5so-0XAvXlWzpKkvsergw6907Vb7JE"
!mkdir ./data/
!mv v4*.zip ./data/
%cd ./data/
!unzip v4*.zip
%cd ..

# Ajusta nomes das pastas onde estão as imagens de treino e teste.
# As imagens de validação serão criadas através de um percentual das
# imagens de treino (como fazemos em compara_classificadores_tf2)
pasta_base = "./"  # No desktop a pasta base é a mesma onde está este código 

# Se for usar o seu google drive, descomente a linha abaixo e ajuste se necessário
#pasta_base = "/content/drive/MyDrive/"


pasta_data = pasta_base+"data/"
print("Vai ler as imagens de: ",pasta_data)
pasta_treino = pasta_data+"train"
pasta_teste  = pasta_data+"test"

# Define as transformações nas imagens:
# Muda tamanho, transforma em tensor e normaliza usando os valores
# calculados sobre a base ImageNet (por conta da transferência de aprendizagem)
#
# Nem sempre a normalização dá melhores resultados. Descomente a linha se quiser usar.
transform = transforms.Compose([transforms.Resize((tamanho_imagens,tamanho_imagens)),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ])
# Prepara banco de imagens de treino (está junto com validação por enquanto)
training_val_data = datasets.ImageFolder(root=pasta_treino,transform=transform)
# Prepara banco de imagens de teste
test_data = datasets.ImageFolder(root=pasta_teste,transform=transform)

# Aqui vai separar em treinamento e validação
train_idx, val_idx = train_test_split(list(range(len(training_val_data))), test_size=perc_val)
training_data = Subset(training_val_data, train_idx)
val_data = Subset(training_val_data, val_idx)


# Cria os objetos que irão manipular os dados (basicamente ajuda a pegar
# lote (batch) de imagens de treinamento e de validação)
train_dataloader = DataLoader(training_data, batch_size=tamanho_lote,shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=tamanho_lote,shuffle=True)

# Mostra informações do primeiro lote de imagens de validação
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

total_imagens=len(training_data)+len(val_data)+len(test_data)
print(f"Total de imagens: {total_imagens}")
print(f"Total de imagens de treinamento: {len(training_data)} ({100*len(training_data)/total_imagens:>2f}%)")
print(f"Total de imagens de validação: {len(val_data)} ({100*len(val_data)/total_imagens:>2f}%)")
print(f"Total de imagens de teste: {len(test_data)} ({100*len(test_data)/total_imagens:>2f}%)")
labels_map = {v: k for k, v in test_data.class_to_idx.items()}
print('\nClasses:',labels_map)

"""### Mostrando algumas imagens"""

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
    # Tem que ajustar a ordem das dimensões do tensor para que os canais
    # fiquem na última dimensão (e não ma primeira)
    plt.imshow(img.permute(1,2,0))

plt.show() # Este é o comando que vai mostrar as imagens

"""## Definindo uma rede neural artificial"""

# Verifica se tem GPU na máquina, caso contrário, usa a CPU mesmo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device}")

# Vai precisar do total de classes para ajustar a última camada
# das redes

total_classes = len(labels_map)
# Inicia com a rede escolhida (acrescente mais "ifs" para outras redes)
# Carrega os pesos pré-treinados na ImageNet (transfer learning)
# Ajusta a última camada para poder corresponder ao total de classes do
# problema atual.
if nome_rede == "resnet":
   model = models.resnet18(pretrained=True)
   model.fc = nn.Linear(model.fc.in_features, total_classes )
elif nome_rede == "squeezenet":
   model = models.squeezenet1_0(pretrained=True)
   model.classifier[1] = nn.Conv2d(512, total_classes, kernel_size=(1,1), stride=(1,1))
   model.num_classes = total_classes
elif nome_rede == "densenet":
   model = models.densenet161(pretrained=True)
   model.classifier = nn.Linear(model.classifier.in_features,total_classes)


# Prepara a rede para o dispositivo que irá processá-la
model = model.to(device)

# Imprime dados sobre a arquitetura da rede
print(model)

# Define o otimizador como sendo descida de gradiente estocástica
otimizador = torch.optim.SGD(model.parameters(), lr=taxa_aprendizagem, momentum=momento)

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

        train_loss += loss.item() # Guarda para calcular a perda média
        # Calcula os acertos para o lote inteiro de imagens
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()


        loss.backward()        # Calcula os gradientes com base no erro (loss)
        optimizer.step()       # Ajusta os pesos com base nos gradientes
        optimizer.zero_grad()  # Zera os gradientes pois vai acumular para todas
                               # as imagens do lote

        # Imprime informação a cada 4 lotes processados
        if batch % 4 == 0:
            # Mostra a perda e o total de imagens já processadas
            loss, current = loss.item(), batch * len(X)
            print(f"Perda Treino: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
      # Salva a melhor rede encontrada até o momento
      torch.save(model.state_dict(), pasta_data+"modelo_treinado_"+nome_rede+".pth")
      print("Salvou o modelo com a maior acurácia na validação até agora em "+pasta_data+"modelo_treinado_"+nome_rede+".pth")
      maior_acuracia = val_acuracia
      total_sem_melhora = 0
    else:
      total_sem_melhora += 1
      print(f"Sem melhora há {total_sem_melhora} épocas ({100*val_acuracia}% <= {100*(maior_acuracia+tolerancia)}%)")
    if total_sem_melhora > paciencia:
      print(f"Acabou a paciência com {epoca+1} épocas ")
      break

print("Terminou a fase de aprendizagem !")

# Pega algumas imagens para o tensorboard mostrar depois
images, labels = next(iter(train_dataloader))
images = images.to(device)
labels = labels.to(device)
# Cria uma grade de imagens para o tensorboard
img_grid = torchvision.utils.make_grid(images)
writer.add_image('Minhas Imagens', img_grid)
writer.add_graph(model, images)
writer.close()

"""## Visualização usando Tensorboard

"""

##Retirar os comentários se quiser usar o tensorboard
#%load_ext tensorboard
#%tensorboard --logdir=runs

"""## Carregando a rede treinada anteriormente e usando


"""

model.load_state_dict(torch.load(pasta_data+"modelo_treinado_"+nome_rede+".pth"))

"""## Usando a rede treinada para classificar algumas imagens"""

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

# Vai mostrar a classificação da rede para 16 imagens do conjunto de teste
figure = plt.figure(figsize=(8, 8))  # Cria o local para mostrar as imagens
cols, rows = 4, 4  # Irá mostrar 16 imagens em uma grade 4x4
print(f"Testando em {len(test_data)} imagens. Resultados:")
for i in range(cols*rows):
    aleatoria = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[aleatoria]

    img = img.unsqueeze(0).to(device)

    # Classifica a imagem usando a rede treinada
    predita = classifica_uma_imagem(model,img,label)
    # Adiciona a imagem na grade que será mostrada
    figure.add_subplot(rows, cols, i+1)
    # Usa a classe da imagem como título da imagem
    plt.title(predita)
    # Não mostra valores para os eixos X e Y
    plt.axis("off")
    # Primeiro converte o tensor para o formato de cpu (o imshow não vai usar GPU)
    # Depois retira a dimensão relacionada com os lotes (dimensão 0)
    # e por fim, faz uma permutação das dimensões de forma que
    # a dimensão relacionada aos canais (RGB) fiquem por último e não em primeiro
    # fiquem na última dimensão (e não ma primeira)
    plt.imshow(img.cpu().squeeze(0).permute(1,2,0))

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
      img = img.unsqueeze(0).to(device)
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

# Normaliza a matriz para o intervalo 0 e 1 e arredonda em 2 casas decimais
# cada célula
matriz_normalizada = np.round(matriz/np.sum(matriz),2)
# Transforma a matriz no formato da biblioteca PANDA
df_matriz = pd.DataFrame(matriz_normalizada, index = classes,
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
