# -*- coding: utf-8 -*-
"""exemplo_pytorch_v5.ipynb


## Tutorial Introdutório de Pytorch (v5)

Tutorial baseado no código disponibilizado por Sagi Eppel [aqui](https://towardsdatascience.com/train-neural-net-for-semantic-segmentation-with-pytorch-in-50-lines-of-code-830c71a6544f)

Principais funcionalidades:

- Exemplo de segmentação semântica usando o pytorch
- Transferência de aprendizado usando a DeepLabV3 pré-treinada
- Informações de desempenho da segmentação (acurácia pixel a pixel e matriz 
  de confusão)

Imagens para treinamento, validação e teste:

- Precisam estar dentro de uma pasta no seu drive chamada data/imagens/

Anotações para treinamento, validação e teste:

- Precisam estar dentro de uma pasta no seu drive chamada data/anotacoes/
- Precisam ter o mesmo nome da imagem correspondente que está na pasta de imagens, mas com extensão .png
- Precisa ser uma imagem em tons de cinza (1 canal) com o valor de pixel correspondente a cada classe do problema. Por exemplo: 0 = Fundo, 1 = Serpente
- O código está assumindo apenas duas classes e realiza uma binarização na imagem de anotação
- Tem um script chamado convertePoligonoPNG, que está na pasta "data" do projeto, que pega anotações de polígonos feitos no Labelme e converte para o formato usado neste exemplos (máscaras em arquivos PNG).

## Carregando um banco de imagens
"""

import torch   # Pytorch principal
from torch import nn  # Módulo para redes neurais (neural networks)
import os      # Funções para manipulação de pastas e arquivos
import numpy as np    # Várias funções numéricas
import torchvision.models.segmentation # Redes famosas para segmentação semântica
import torchvision.transforms as transforms
import matplotlib.pyplot as plt # Mostra imagens e gráficos
from torch.utils.tensorboard import SummaryWriter # Salva "log" da aprendizagem
import torchvision
from PIL import Image,ImageOps
import torch.utils.data as data
import sklearn.metrics as metrics  # Ajuda a calcular métricas de desempenho
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
import seaborn as sn  # Usado para gerar um mapa de calor para a matriz de confusão
import pandas as pd   # Ajuda a trabalhar com tabelas
import numpy as np    # Várias funções numéricas


# Definindo alguns hiperparâmetros importantes:
epocas = 100  # Total de passagens durante a aprendizagem pelo conjunto de imagens
tamanho_lote = 2  # Tamanho de cada lote sobre o qual é calculado o gradiente
taxa_aprendizagem = 0.1   # Magnitude das alterações nos pesos
momento = 0.1  # Mantem informação de pesos anteriores (as mudanças de
               # de peso passam a ser mais suaves). Não é usado no
               # otimizador ADAM, apenas no SGD.
paciencia = 10  # Total de épocas sem melhoria da acurácia na validação até parar
tolerancia = 0.01 # Melhoria menor que este valor não é considerada melhoria
perc_teste = 0.2  # Percentual a ser usado para teste
perc_val = 0.3    # Percentual do treinamento a ser usado para validação

# Define uma arquitetura já conhecida que será usada
# Opções atuais: "deeplabv3","fcn" 
nome_rede = "fcn"
tamanho_imagens = 500  # Tamanho das imagem para a arquitetura escolhida

# Lista de classes 
classes=['fundo','cascavel']

# Descomente o código abaixo se quiser montar e usar o seu próprio google drive
# no lugar das pastas que o colab cria automaticamente 
# from google.colab import drive
# drive.mount('/content/drive')

# Vai baixar o banco de imagens de treino e de teste do exemplo com serpentes
!curl -L -o v5_imagens_anotacoes.zip "https://drive.google.com/uc?export=download&id=1B1IqZ4oHWH_-CRpRVG1MExjziBn2SMOC"
!mkdir ./data/
!mv v5*.zip ./data/
%cd ./data/
!unzip v5*.zip
%cd ..


# Ajusta nome das pastas onde estão todas as imagens e anotações
pasta_data = "./data/"  

# Se estive usando seu próprio drive, descomente e ajuste a linha abaixo
#pasta_data = "/content/drive/MyDrive/data/"

print("Vai ler as imagens de: ",pasta_data)

# Define as transformações nas imagens: 
# Muda tamanho e transforma em tensor 
transform = transforms.Compose([transforms.Resize((tamanho_imagens,tamanho_imagens)),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ])    

# Lê uma imagem aleatória do disco e sua anotação
def ImagemAleatoria(pasta,nomes):
    idx=np.random.randint(0,len(nomes)) # Seleciona um índice aleatório
    
    # Lê a imagem usando o índice aleatório
    imagem=Image.open(os.path.join(pasta, "imagens", nomes[idx]))

    # Aplica operações para resolver problema de orientação
    # de imagens com tag de orientação EXIF
    imagem = ImageOps.exif_transpose(imagem)

    # Lê a anotação usando o índice aleatório. 
    # Assume que as anotações são do tipo png
    arquivo_anotacao=os.path.splitext(nomes[idx])[0]+'.png'  
    anotacao=Image.open(os.path.join(pasta, "anotacoes", arquivo_anotacao))
    # Aplica as transformações
    imagem=transform(imagem)
    anotacao=transform(anotacao)
    # Binariza a anotação
    # Dependendo do tipo do arquivo de anotação pode ser preciso mudar
    # o teste "anotacao > 0". Tem que dar uma inspecionada nos valores dos pixels
    anotacao=torch.where(anotacao > 0, 1, 0)

    return imagem,anotacao

# Lê um lote de imagens
def LoteDeImagens(pasta,nomes,tamanho_lote): 
    imagens = torch.zeros([tamanho_lote,3,tamanho_imagens,tamanho_imagens])
    anotacoes = torch.zeros([tamanho_lote,tamanho_imagens,tamanho_imagens])
    
    for i in range(tamanho_lote):
        imagens[i],anotacoes[i]=ImagemAleatoria(pasta,nomes)
    
    return imagens, anotacoes

# Cria uma lista com os nomes das imagens que estão na pasta de treino 
nomes_todas=os.listdir(os.path.join(pasta_data, "imagens")) 
# Divide as imagens que estavam na pasta treino entre treino e validação
other_idx,test_idx = train_test_split(list(range(len(nomes_todas))), test_size=perc_teste)
nomes_teste = [nomes_todas[i] for i in test_idx]
nomes_other = [nomes_todas[i] for i in other_idx]
train_idx, val_idx = train_test_split(list(range(len(nomes_other))), test_size=perc_val)
nomes_treino = [nomes_other[i] for i in train_idx]
nomes_val = [nomes_other[i] for i in val_idx]

print('Treino:',nomes_treino)
print('Validação:',nomes_val)
print('Teste:',nomes_teste)

# Carrega um lote de imagens e de anotações de treino
X,y = LoteDeImagens(pasta_data,nomes_treino,tamanho_lote)

# Mostra informações de um lote de imagens de validação 
# X vai conter um lote de imagens
# y vai conter as anotações
print(f"Tamanho do lote de imagens: {X.shape[0]}")
print(f"Quantidade de canais: {X.shape[1]}")
print(f"Altura de cada imagem: {X.shape[2]}")
print(f"Largura de cada imagem: {X.shape[3]}")
print(f"Tamanho do lote de anotações: {y.shape[0]}")

total_imagens=len(nomes_treino)+len(nomes_val)+len(nomes_teste)
print(f"Total de imagens: {total_imagens}")
print(f"Total de imagens de treinamento: {len(nomes_treino)} ({100*len(nomes_treino)/total_imagens:>2f}%)")
print(f"Total de imagens de validação: {len(nomes_val)} ({100*len(nomes_val)/total_imagens:>2f}%)")
print(f"Total de imagens de teste: {len(nomes_teste)} ({100*len(nomes_teste)/total_imagens:>2f}%)")

print('Classes: ',classes,'Total = ',len(classes))

"""### Mostrando algumas imagens"""

figure = plt.figure(figsize=(10, 5))  # Cria o local para mostrar as imagens
# Não mostra valores para os eixos X e Y
plt.axis("off")
cols, rows = 4, 2  # Irá mostrar 2 imagens com suas anotações em uma grade 4x1

# Carrega um lote de imagens e de anotações de treino
X,y = LoteDeImagens(pasta_data,nomes_treino,tamanho_lote)

# Passa por cada imagem do lote
for i in range(0,len(X)):
    # Pega um imagem e sua anotação
    imagem = X[i]
    anotacao = y[i]

    # Adiciona a imagem na grade que será mostrada
    # Tem que ajustar a ordem das dimensões do tensor para que os canais
    # fiquem na última dimensão (e não ma primeira)
    figure.add_subplot(rows, cols, i*2+1)
    plt.imshow(imagem.permute(1,2,0),origin='lower')
    # Adiciona anotação ao lado da imagem
    figure.add_subplot(rows, cols, i*2+2)
    plt.imshow(anotacao.squeeze().numpy(),cmap='gray',origin='lower')
    
plt.show() # Este é o comando que vai mostrar as imagens

"""## Definindo uma rede neural artificial"""

# Verifica se tem GPU na máquina, caso contrário, usa a CPU mesmo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device}")

# Inicia com a rede escolhida (acrescente mais "ifs" para outras redes)
# Carrega os pesos pré-treinados na ImageNet (transfer learning)
# Ajusta a última camada para poder corresponder ao total de classes do
# problema atual. 
if nome_rede == "deeplabv3":
   model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
   # Muda a camada final para um problema com 2 classes
   model.classifier[4] = torch.nn.Conv2d(256, len(classes), kernel_size=(1, 1), stride=(1, 1)) 
elif nome_rede == "fcn":
   model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
   model.classifier[4] = torch.nn.Conv2d(512, len(classes), kernel_size=(1, 1), stride=(1, 1)) 
   #model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, num_classes=len(classes))
        

# Prepara a rede para o dispositivo que irá processá-la
model = model.to(device)

# Imprime dados sobre a arquitetura da rede
print(model)

# Define o otimizador como sendo o de estimativa adaptativa do momento (ADAM)
otimizador = torch.optim.Adam(model.parameters(), lr=taxa_aprendizagem)
# Descomente a linha de baixo se quiser usar descida de gradiente estocástica (SGD)
# otimizador = torch.optim.SGD(model.parameters(), lr=taxa_aprendizagem, momentum=momento)

# Define a função de perda como entropia cruzada
funcao_perda = nn.CrossEntropyLoss()

# Cria o módulo do tensorboard de coleta de dados
writer = SummaryWriter()

# Define a função para treinar a rede
# dataloader = módulo que manipula o conjunto de imagens
# model = arquitetura da rede
# loss_fn = função de perda
# optimizer = otimizador 
def train(pasta, nomes, model, loss_fn, optimizer):


    size = len(nomes)  # Total de imagens
    num_batches = int(size/tamanho_lote)   # Total de lotes
    pixels = num_batches*tamanho_lote*tamanho_imagens*tamanho_imagens

    model.train()  # Avisa que a rede vai entrar em modo de aprendizagem

    train_loss, train_correct = 0, 0  # Usado para calcular perda e acurácia médias

    # Pega um lote de imagens de cada vez do conjunto de treinamento
    for batch in range(0,num_batches):

        X,y = LoteDeImagens(pasta,nomes,tamanho_lote)
        X, y = X.to(device), y.to(device)  # Prepara os dados para o dispositivo (GPU ou CPU)
        pred = model(X)['out']    # Realiza uma previsão usando os pesos atuais
        loss = loss_fn(pred, y.long())  # Calcula o erro com os pesos atuais

        train_loss += loss.item() # Guarda para calcular a perda média
        # Calcula os acertos para o lote inteiro de imagens
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item() 

        loss.backward()        # Calcula os gradientes com base no erro (loss)
        optimizer.step()       # Ajusta os pesos com base nos gradientes
        optimizer.zero_grad()  # Zera os gradientes pois vai acumular para todas
                               # as imagens do lote

        # Imprime informação a cada 1 lote processado
        if batch % 1 == 0:
            # Mostra a perda e o total de imagens já processadas
            loss, current = loss.item(), batch * len(X)
            print(f"Perda Treino: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches  # Como a perda foi calculada por lote, divide
                               # pelo total de lotes para calcular a média
    train_acuracia = train_correct / pixels # Já o total de acertos é em relação
                                            # ao total geral de pixels

    return train_loss, train_acuracia        

# Define a função de validação (aqui a rede não está aprendendo, apenas
# usando "aquilo que aprendeu", mas em um conjunto de imagens diferente
# do conjunto usado para aprender)
def validation(pasta, nomes, model, loss_fn):


    size = len(nomes)  # Total de imagens
    print('Total de imagens:',size)
    num_batches = int(size/tamanho_lote)   # Total de lotes
    print('Total de lotes:',num_batches)
    pixels = num_batches*tamanho_lote*tamanho_imagens*tamanho_imagens
    model.eval()  # Avisa que a rede vai entrar em modo de aprendizagem


    # Vai calcular a perda e o total de acertos no conjunto de validação
    val_loss, val_correct = 0, 0

    # Na validação os pesos não são ajustados e por isso não precisa
    # calcular o gradiente
    with torch.no_grad():
        # Pega um lote de imagens de cada vez do conjunto de treinamento
        for batch in range(0,num_batches):

            X,y = LoteDeImagens(pasta,nomes,tamanho_lote)
            X, y = X.to(device), y.to(device)  # Prepara os dados para o dispositivo (GPU ou CPU)
            pred = model(X)['out']    # Realiza uma previsão usando os pesos atuais
            val_loss += loss_fn(pred, y.long()).item()
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    val_acuracia = val_correct / pixels

    print("Informações na Validação:")
    print(f"Total de acertos: {int(val_correct)}")
    print(f"Total de pixels: {pixels}")
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
    train_loss, train_acuracia = train(pasta_data,nomes_treino, model, funcao_perda, otimizador)
    val_loss, val_acuracia = validation(pasta_data,nomes_val, model, funcao_perda)

    # Guarda informações para o tensorboard pode criar os gráficos depois
    writer.add_scalars('Loss', {'train':train_loss,'val':val_loss}, epoca)
    writer.add_scalars('Accuracy', {'train':train_acuracia,'val':val_acuracia}, epoca)

    # Soma uma tolerancia no valor da maior acurácia para que melhoras muito
    # pequenas não sejam consideradas
    if val_acuracia > (maior_acuracia+tolerancia): 
      # Salva a melhor rede encontrada até o momento
      torch.save(model.state_dict(), pasta_data+"modelo_treinado_"+nome_rede+".pth")
      print("Salvou o modelo com a maior acurácia na validação até agora em modelo_treinado_"+nome_rede+".pth")      
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
# images,anotacoes = LoteDeImagens(pasta_data,nomes_treino,tamanho_lote)
# images = images.to(device)
# anotacoes = anotacoes.to(device)
# Cria uma grade de imagens para o tensorboard
# img_grid = torchvision.utils.make_grid(images)
# writer.add_image('Minhas Imagens', img_grid)
# writer.add_graph(model, images)
writer.close()

"""## Visualização usando Tensorboard

"""

##Retirar os comentários se quiser usar o tensorboard
#%load_ext tensorboard
#%tensorboard --logdir=runs

"""## Carregando a rede treinada anteriormente e usando


"""

model.load_state_dict(torch.load(pasta_data+"modelo_treinado_"+nome_rede+".pth"))

"""## Usando a rede treinada para segmentar algumas imagens """

# Classifica uma única imagem 
# model: rede a ser usada
# x: imagem
# y: classificação real da imagem
# predita: classificação dada pela rede
def classifica_uma_imagem(model,x):
    model.eval()
    x = x.to(device) 

    with torch.no_grad():
       predita = model(torch.unsqueeze(x, dim=0))['out']

    return(predita)


figure = plt.figure(figsize=(12, 8))  # Cria o local para mostrar as imagens
# Não mostra valores para os eixos X e Y
plt.axis("off")
cols, rows = 4, 4  # Irá mostrar imagens com suas anotações em uma grade 4x2

# Carrega um lote de imagens e de anotações de teste
X,y = LoteDeImagens(pasta_data,nomes_teste,tamanho_lote)

# Passa por cada imagem do lote
for i in range(0,len(X)):
    # Pega um imagem e sua anotação
    imagem = X[i]
    anotacao = y[i]

    # Classifica a imagem usando a rede treinada
    predita = classifica_uma_imagem(model,imagem)
    pixels = tamanho_imagens*tamanho_imagens

    corretos = (predita.argmax(1) == anotacao.to(device)).type(torch.float).sum().item()
    print('Corretos na imagem:',corretos)
    print('Total pixels:', pixels)
    print('Acurácia na imagem, pixel a pixel:',corretos/pixels)

    predita = predita.argmax(1).squeeze()
    # Adiciona a imagem na grade que será mostrada
    # Tem que ajustar a ordem das dimensões do tensor para que os canais
    # fiquem na última dimensão (e não ma primeira)
    figure.add_subplot(rows, cols, i*4+1)
    plt.imshow(imagem.permute(1,2,0))
    # Adiciona anotação real ao lado da imagem
    figure.add_subplot(rows, cols, i*4+2)
    plt.imshow(anotacao.squeeze().numpy(),cmap='gray',origin='upper')
    # Adiciona anotação predita ao lado da anotacao real
    figure.add_subplot(rows, cols, i*4+3)
    plt.imshow(predita.cpu().numpy(),cmap='gray',origin='upper')
    # Adiciona a máscara predita sobreposta à imagem original
    sobreposta = imagem.cpu() * predita.cpu()
    figure.add_subplot(rows, cols, i*4+4)
    plt.imshow(sobreposta.permute(1,2,0))
    
plt.show() # Este é o comando que vai mostrar as imagens

"""## Gerando algumas estatísticas no conjunto de teste"""

# Listas para guardar valores preditos e reais
predicoes = []
reais = []

size = len(nomes_teste)  # Total de imagens
num_batches = int(size/tamanho_lote)   # Total de lotes
pixels = num_batches*tamanho_lote*tamanho_imagens*tamanho_imagens

# Vai acumular acertos para calcular acurácia
test_correct=0

model.eval() # Coloca a rede no modo de avaliação (e não de aprendizagem)
with torch.no_grad():   # Avisa que não devem ser calculados gradientes
   for batch in range(0,num_batches):
      X,y = LoteDeImagens(pasta_data,nomes_teste,tamanho_lote)
      X, y = X.to(device), y.to(device)  # Prepara os dados para o dispositivo (GPU ou CPU)
      predicao = model(X)['out']    # Realiza uma previsão usando os pesos atuais
      test_correct += (predicao.argmax(1) == y).type(torch.float).sum().item()     
      predicao = predicao.argmax(1)  # Pega a classe com maior valor

      predicoes.extend(predicao.flatten().tolist()) # Guarda predição na lista
      reais.extend(y.flatten().tolist())        # Guarda valor real na lista

# Acurácia no conjunto de teste
test_acuracia = test_correct/pixels

# Constroi a matriz de confusão
matriz = metrics.confusion_matrix(reais,predicoes)

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
print(f'Resumo para as {len(nomes_teste)} imagens de teste:')
print(f"Acertos: {int(test_correct)}")
print(f"Acurácia: {(100*test_acuracia):>0.2f}%")
print(f"Precisão: {100*precision:>0.2f}%")
print(f"Revocação: {100*recall:>0.2f}%")
print(f"Medida-F: {100*fscore:>0.2f}%")
print('-----------------------------------')
