# -*- coding: utf-8 -*-
"""exemplo_pytorch_v6.ipynb

## Tutorial Introdutório de Pytorch (v6)

Tutorial baseado, mas com muitas modificações, no código disponibilizado por [Sovit Ranjan Rath](https://debuggercafe.com/a-simple-pipeline-to-train-pytorch-faster-rcnn-object-detection-model/)

Principais funcionalidades:

- Exemplo de detecção de objetos
- Transferência de aprendizado usando a Faster RCNN pré-treinada
- Aumento de dados


Imagens e anotações para treinamento, validação e teste:

- Precisam estar dentro de uma pasta no seu drive chamada data/condensadores
- As anotações devem seguir o formato Pascal VOC, com um arquivo .xml para cada imagem .jpg

## Carregando o banco de imagens
"""

import torch   # Pytorch principal
from torch import nn  # Módulo para redes neurais (neural networks)
import os,fnmatch      # Funções para manipulação de pastas e arquivos
import numpy as np    # Várias funções numéricas
import torchvision
import torchvision.models.segmentation # Redes famosas para segmentação semântica
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt # Mostra imagens e gráficos
from torch.utils.tensorboard import SummaryWriter # Salva "log" da aprendizagem
from torch.utils.data import Dataset, DataLoader
import cv2  # Biblioteca OpenCV para manipular imagens
import albumentations as A  # Biblioteca com diversos tipos de transformações
                            # para imagens
from albumentations.pytorch import ToTensorV2                            
from xml.etree import ElementTree as et # Manipulação de arquivos XML
import torch.utils.data as data
import sklearn.metrics as metrics  # Ajuda a calcular métricas de desempenho
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
import seaborn as sn  # Usado para gerar um mapa de calor para a matriz de confusão
import pandas as pd   # Ajuda a trabalhar com tabelas
import numpy as np    # Várias funções numéricas


# Definindo alguns hiperparâmetros importantes:
epocas = 100  # Total de passagens durante a aprendizagem pelo conjunto de imagens
tamanho_lote = 4  # Tamanho de cada lote sobre o qual é calculado o gradiente
taxa_aprendizagem = 0.01   # Magnitude das alterações nos pesos
momento = 0.1  # Mantem informação de pesos anteriores (as mudanças de
               # de peso passam a ser mais suaves). Não é usado no
               # otimizador ADAM, apenas no SGD.
peso_regularizador = 0 # Peso do regularizador, geralmente norma L2,
                            # que é adicionado à função de perda (weigth_decay)
paciencia = 10  # Total de épocas sem melhoria da acurácia na validação até parar
tolerancia = 0.001 # Melhoria menor que este valor não é considerada melhoria
perc_teste = 0.2  # Percentual a ser usado para teste
perc_val = 0.3    # Percentual do treinamento a ser usado para validação

# Define uma arquitetura já conhecida que será usada
# Opções atuais: "fasterRCNN"
nome_rede = "faster"
largura_imagens = 416  # Largura das imagem para a arquitetura escolhida
altura_imagens = 416  # Altura das imagem para a arquitetura escolhida

# Lista de classes. Tem que colocar sempre a classe fundo.
classes=['fundo','conde']

# Pasta onde estão os dados para treinamento e teste
pasta_data = "./data/condensadores/"  
print("Vai ler as imagens de: ",pasta_data)

# Se for usar seu próprio Drive descomente e ajuste a linha abaixo
#pasta_data = "/content/drive/MyDrive/data/condensadores/"


# Descomente o código abaixo se quiser montar e usar o seu próprio google drive
# no lugar das pastas que o colab cria automaticamente 
# from google.colab import drive
# drive.mount('/content/drive')

# Vai baixar o banco de imagens de treino e de teste do exemplo com serpentes
!curl -L -o v6_condensadores.zip "https://drive.google.com/uc?export=download&id=1UY4906H7PsB7KXFG-v1ADupSQXJXxQK7"   
!mkdir ./data/
!mv v6*.zip ./data/
%cd ./data/
!unzip v6*.zip
%cd ..



# Verifica se tem GPU na máquina, caso contrário, usa a CPU mesmo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device}")

"""### Definindo a classe CustomDataset

Diferentemente dos exemplos até agora, desta vez iremos criar uma classe para tratar o nosso banco de imagens pois as anotações são um pouco mais complexas.
"""

# Cria uma classe para tratar o meu próprio banco de imagens (Custom) tendo
# como base a classe "Dataset" do pytorch
class CustomDataset(Dataset):
    def __init__(self, pasta, nomes_arquivos, largura, altura, classes, transformacoes=None):
        self.pasta = pasta  # Pasta onde estão as imagens
        self.nomes_arquivos = nomes_arquivos  # Nomes das imagens a serem utilizadas
        self.altura = altura   # Altura que as imagens deverão ter
        self.largura = largura    # Largura que as imagens deverão ter 
        self.classes = classes  # Nomes das classes do problema
        self.transformacoes = transformacoes # Transformações a serem aplicadas      
    def __getitem__(self, idx):
        # Lê uma imagem com o OpenCV
        nome_imagem = self.nomes_arquivos[idx]
        imagem = cv2.imread(os.path.join(self.pasta,nome_imagem))
        # Converte de BGR, que usado pelo OpenCV, para RGB 
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Redimensiona a imagem
        redimensionada = cv2.resize(imagem, (self.largura, self.altura))
        # Coloca os valores dos pixels no intervalo [0,1]
        redimensionada /= 255.0
        
        # Troca a extensão do arquivo para poder pegar as anotações em xml
        nome_anotacao = nome_imagem[:-4] + '.xml'
        xml_anotacao = os.path.join(self.pasta, nome_anotacao)
        
        # Vai guardar os retângulos e classe de cada retângulo
        boxes = []
        labels = []
        # Lê o arquivo XML
        tree = et.parse(xml_anotacao)
        root = tree.getroot()
        
        # Altura e Largura original da imagem
        altura = imagem.shape[0]
        largura = imagem.shape[1]
        
        # Vai ler as coordenadas dos retângulos de anotações e ajustar para
        # o novo tamanho da imagem (usa vários comandos da biblioteca xml.etree
        # para isso)
        for member in root.findall('object'):
            # Pega a classe da anotação (neste exemplo, "conde" (condenador))
            labels.append(self.classes.index(member.find('name').text))
            
            # Pega as coordenadas das extremidades do retângulo
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # Redimensiona os retângulos de anotação         
            xmin_final = (xmin/largura)*self.largura
            xmax_final = (xmax/largura)*self.largura
            ymin_final = (ymin/altura)*self.altura
            ymax_final = (ymax/altura)*self.altura

            # Pelo arredondamento, na hora de converter os retângulos, 
            # pode passar da largura ou da altura da imagem. O código
            # abaixo verifica e corrige isso.
            if xmax_final > self.largura: xmax_final = self.largura
            if ymax_final > self.altura: ymax_final = self.altura
            if xmin_final < 0: xmin_final = 0
            if ymin_final < 0: ymin_final = 0                       
          
            # Guarda o retângulo na lista de retângulos
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Converte os retângulo para um tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Calcula a área do retângulo
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Avisa que não é um problema de detecção de multidões
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Converte as classes para um tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Cria um dicionário com todas as informações que a rede vai
        # precisar depois
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # Aplica as transformações que foram passadas como parâmetro
        if self.transformacoes:
            sample = self.transformacoes(image = redimensionada,
                                     bboxes = target['boxes'],
                                     labels = labels)
            redimensionada = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

            
        return redimensionada, target

    def __len__(self):
        return len(self.nomes_arquivos)

"""### Criando os bancos de treino, validação e teste"""

# Cria uma lista com os nomes de todas imagens disponíveis
nomes_todas=fnmatch.filter(os.listdir(pasta_data), "*.jpg")


# Dividirá as imagens entre treino, validação e teste
# Primeiro separa o teste
other_idx,test_idx = train_test_split(list(range(len(nomes_todas))), test_size=perc_teste)
nomes_teste = [nomes_todas[i] for i in test_idx]
nomes_other = [nomes_todas[i] for i in other_idx]
# E depois separa entre treino e validação
train_idx, val_idx = train_test_split(list(range(len(nomes_other))), test_size=perc_val)
nomes_treino = [nomes_other[i] for i in train_idx]
nomes_val = [nomes_other[i] for i in val_idx]

# Mostra os nomes das imagens de treino, validação e teste
print('Treino:',nomes_treino)
print('Validação:',nomes_val)
print('Teste:',nomes_teste)

# Mostra totais e percentuais de treino, validação e teste
total_imagens=len(nomes_treino)+len(nomes_val)+len(nomes_teste)
print(f"Total de imagens: {total_imagens}")
print(f"Total de imagens de treinamento: {len(nomes_treino)} ({100*len(nomes_treino)/total_imagens:>2f}%)")
print(f"Total de imagens de validação: {len(nomes_val)} ({100*len(nomes_val)/total_imagens:>2f}%)")
print(f"Total de imagens de teste: {len(nomes_teste)} ({100*len(nomes_teste)/total_imagens:>2f}%)")

# Mostra os nomes e o total de classes
print('Classes: ',classes,'Total = ',len(classes))

# Cria o objeto que vai representar os bancos de treino validação e teste
# usando a classe CustomDataset criada anteriormente
# Aplica aumento de dados no treinamento com flip, rotação e 3 tipos de suavização
# Usa uma biblioteca chamada Albumentation para fazer isso (A.)
treino = CustomDataset(pasta_data,nomes_treino,largura_imagens,altura_imagens,classes,
                       A.Compose([
                                      A.Flip(0.5),  
#                                     A.RandomRotate90(0.5),
#                                     A.MotionBlur(p=0.2),
#                                     A.MedianBlur(blur_limit=3, p=0.1),
#                                     A.Blur(blur_limit=3, p=0.1),
                                     ToTensorV2(p=1.0)
                                 ], bbox_params={
                                      'format': 'pascal_voc',
                                      'label_fields': ['labels']
                                 }))

val = CustomDataset(pasta_data,nomes_val,largura_imagens,altura_imagens,classes,
                       A.Compose([
                                    ToTensorV2(p=1.0)
                                 ], bbox_params={
                                      'format': 'pascal_voc', 
                                      'label_fields': ['labels']
                                 }))

teste = CustomDataset(pasta_data,nomes_teste,largura_imagens,altura_imagens,classes,
                       A.Compose([
                                    ToTensorV2(p=1.0)
                                 ], bbox_params={
                                      'format': 'pascal_voc', 
                                      'label_fields': ['labels']
                                 }))

# Ajusta os dados para quando o número de objetos em cada imagem
# é diferente
def collate_fn(lote):
    return tuple(zip(*lote))

# Cria os objetos para carregar lotes de imagens e anotações para treino
lote_treino = DataLoader(
        treino,
        batch_size=tamanho_lote,
        shuffle=True,
        collate_fn=collate_fn
    )

# Cria os objetos para carregar lotes de imagens e anotações para validação
lote_val = DataLoader(
        val,
        batch_size=tamanho_lote,
        shuffle=True,
        collate_fn=collate_fn
    )


# Cria os objetos para carregar lotes de imagens e anotações para treino
lote_teste = DataLoader(
        teste,
        batch_size=tamanho_lote,
        shuffle=True,
        collate_fn=collate_fn
    )

"""### Mostrando algumas imagens"""

# Vai colocar os retângulos de anotação dentro da imagem
def cria_imagem_anotada(imagem,anotacoes,cor,espessura=2,mostra_texto=False):

    # Pega as coordenadas do retângulo de todas as anotações da imagem
    boxes = anotacoes['boxes'].cpu().numpy().astype(np.int32)
    # Pega as classes de cada anotação
    labels = anotacoes['labels'].cpu().numpy().astype(np.int32)
    # Passa por cada um dos retângulos
    for box_num, box in enumerate(boxes):
        # Desenha o retângulo na imagem
        cv2.rectangle(imagem,
               (box[0], box[1]),
               (box[2], box[3]),
               cor, espessura)
        # Coloca o nome da classe perto do retângulo
        if mostra_texto == True:
               cv2.putText(imagem, classes[labels[box_num]], 
                   (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, cor, 2)
    return imagem

figure = plt.figure(figsize=(8, 8))  # Cria o local para mostrar as imagens
# Não mostra valores para os eixos X e Y
plt.axis("off")
cols, rows = 2, 2  # Irá mostrar 4 imagens com suas anotações em uma grade 2x2

# Pega um lote de imagens com sua estrutura de anotações
images, targets = next(iter(lote_treino))
# Converte as imagens para uso no dispositivo escolhido (GPU ou CPU)
images = list(image.to(device) for image in images)
# Converte as anotações para uso no dispositivo escolhido (GPU ou CPU)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# Laço para pegar 4 imagens do treino
for i in range(0,4):

    # Pega um imagem e suas anotaçãos
    imagem = images[i]
    anotacoes = targets[i]
    
    # Coloca as anotações na imagem para poder mostrar bonitinho
    imagem = cria_imagem_anotada(imagem.permute(1, 2, 0).cpu().numpy(),
                                 anotacoes,(0,0,255))
    # Adiciona a imagem na grade que será mostrada
    figure.add_subplot(rows, cols, i+1)
    plt.imshow(imagem)
    
plt.show() # Este é o comando que vai mostrar as imagens

"""## Definindo uma rede neural artificial"""

# Inicia com a rede escolhida (acrescente mais "ifs" para outras redes)
# Carrega os pesos pré-treinados na ImageNet (transfer learning)
# Ajusta a última camada para poder corresponder ao total de classes do
# problema atual. 
if nome_rede == "faster":
   # Usa um modelo pré-treinado da Faster RCNN com ResNet50 como 
   # espinha dorsal (backbone)
   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   # Pega o total de atributos de entrada da camada de previsão (classificação)
   total_atributos = model.roi_heads.box_predictor.cls_score.in_features
   # Altera a camada que faz a previsão das classes para o total de classes
   # no nosso problema
   model.roi_heads.box_predictor = FastRCNNPredictor(total_atributos, len(classes)) 
elif nome_rede == "retinanet":
   print('Falta implementar')

# Prepara a rede para o dispositivo que irá processá-la
model = model.to(device)

# Imprime dados sobre a arquitetura da rede
print(model)

# Define o otimizador como sendo a descida estocástica de gradiente
otimizador = torch.optim.SGD(model.parameters(), lr=taxa_aprendizagem, 
                                                 momentum=momento,
                                                 weight_decay=peso_regularizador)
  
# Cria o módulo do tensorboard de coleta de dados
writer = SummaryWriter()

# Define a função para treinar a rede
# lotes = módulo que vai fornecer os lotes de imagens e anotações
# model = arquitetura da rede
# optimizer = otimizador 
def train(lotes, model, optimizer):

    size = len(lotes.dataset)  # Total de imagens
    num_batches = int(size/tamanho_lote)   # Total de lotes

    model.train()  # Avisa que a rede vai entrar em modo de aprendizagem

    train_loss = 0  # Usado para calcular perda média

    # Pega um lote de imagens de cada vez do conjunto de treinamento
    for batch, (images, targets) in enumerate(lotes):
    
        # Coloca imagens e anotações no formato necessário (CPU ou GPU)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Realiza a previsão e pega os valores de perda (no caso da detecção
        # nós podemos ter várias funções de perda trabalhando em conjunto)
        loss_dict = model(images, targets)
        loss_sum = sum(loss for loss in loss_dict.values()) # Soma todas as perdas 

        train_loss += loss_sum.item() # Guarda para calcular a perda média

        loss_sum.backward()        # Calcula os gradientes com base no erro (loss)
        optimizer.step()       # Ajusta os pesos com base nos gradientes
        optimizer.zero_grad()  # Zera os gradientes pois vai acumular para todas
                               # as imagens do lote

        # Imprime informação a cada 2 lotes processados
        if (batch) % 2 == 0:
            # Mostra a perda e o total de imagens já processadas          
            print(f"Perda Total no Treino: {loss_sum.item():>7f} [{batch*tamanho_lote:>5d}/{len(lotes.dataset):>5d}]")
            # Mostra cada uma das perdas individualmente
            print('   Por partes: ',[(perda,loss_dict[perda].item()) for perda in loss_dict])


    train_loss /= num_batches  # Como a perda foi calculada por lote, divide
                               # pelo total de lotes para calcular a média

    return train_loss

# Define a função de validação (aqui a rede não está aprendendo, apenas
# usando "aquilo que aprendeu", mas em um conjunto de imagens diferente
# do conjunto usado para aprender)
def validation(lotes, model):

    size = len(lotes.dataset)  # Total de imagens
    num_batches = int(size/tamanho_lote)   # Total de lotes

    #model.eval()  # Avisa que a rede vai entrar em modo de aprendizagem

    val_loss = 0  # Usado para calcular perda média

    # Pega um lote de imagens de cada vez do conjunto de treinamento
    for batch, (images, targets) in enumerate(lotes):
    
        # Coloca imagens e anotações no formato necessário (CPU ou GPU)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Realiza a previsão e pega os valores de perda (no caso da detecção
        # nós podemos ter várias funções de perda trabalhando em conjunto)
        with torch.no_grad():
           loss_dict = model(images, targets)

        loss_sum = sum(loss for loss in loss_dict.values()) # Soma todas as perdas 

        val_loss += loss_sum.item() # Guarda para calcular a perda média


    val_loss /= num_batches  # Como a perda foi calculada por lote, divide
                               # pelo total de lotes para calcular a média

    # Ainda não implementamos métricas de desempenho
    print("Informações na Validação:")
    print(f"===> Perda total média: {val_loss:>8f}")            
    return val_loss

"""## Treinando a Rede Neural (Aprendizagem)"""

# A aprendizagem agora tem parada antecipada (early stopping)
# Diferente de v5, estamos acompanhando a perda e não a acurácia 
menor_perda = 10000  # Guarda a menor perda no conjunto de validação até o momento
total_sem_melhora = 0  # Guarda quantas épocas passou sem melhoria na acurácia

# Passa por todas as imagens várias vezes (a quantidade de vezes
# é definida pelo hiperparâmetro "epocas")
for epoca in range(epocas):

    print(f"-------------------------------")
    print(f"Época {epoca+1} \n-------------------------------")
    train_loss = train(lote_treino, model, otimizador)
    val_loss = validation(lote_val, model)

    # Guarda informações para o tensorboard pode criar os gráficos depois
    writer.add_scalars('Loss', {'train':train_loss,'val':val_loss}, epoca)

    # Diminui uma tolerancia no valor da menor perda para que melhoras muito
    # pequenas não sejam consideradas
    if val_loss < (menor_perda-tolerancia): 
      # Salva a melhor rede encontrada até o momento
      torch.save(model.state_dict(), pasta_data+"modelo_treinado_"+nome_rede+".pth")
      print("Salvou o modelo com a maior acurácia na validação até agora em modelo_treinado_"+nome_rede+".pth")      
      menor_perda = val_loss
      total_sem_melhora = 0
    else: 
      total_sem_melhora += 1 
      print(f"Sem melhora há {total_sem_melhora} épocas ({100*val_loss}% <= {100*(menor_perda-tolerancia)}%)")
    if total_sem_melhora > paciencia:
      print(f"Acabou a paciência com {epoca+1} épocas ")
      break

print("Terminou a fase de aprendizagem !")

writer.close()

"""## Visualização usando Tensorboard

"""

##Retirar os comentários se quiser usar o tensorboard
#%load_ext tensorboard
#%tensorboard --logdir=runs

"""## Carregando a rede treinada anteriormente e usando


"""

model.load_state_dict(torch.load(pasta_data+"modelo_treinado_"+nome_rede+".pth"))

"""## Usando a rede treinada para detectar objetos

Usando o conjunto de teste
"""

figure = plt.figure(figsize=(8, 8))  # Cria o local para mostrar as imagens
# Não mostra valores para os eixos X e Y
plt.axis("off")
cols, rows = 2, 2  # Irá mostrar 4 imagens com suas anotações em uma grade 2x2

# Pega um lote de imagens com sua estrutura de anotações
images, targets = next(iter(lote_teste))
# Converte as imagens para uso no dispositivo escolhido (GPU ou CPU)
images = list(image.to(device) for image in images)
# Converte as anotações para uso no dispositivo escolhido (GPU ou CPU)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# Modo de uso do modelo já treinado
model.eval()

# Realiza a previsão 
with torch.no_grad():
     previsoes = model(images, targets)

# Laço para pegar 4 imagens do treino
for i in range(0,4):

    # Pega um imagem e suas anotaçãos
    imagem = images[i]
    anotacoes = targets[i]
    anotacoes_previstas = previsoes[i]

    print('Imagem ',i+1,' Objetos Reais = ',len(anotacoes['boxes']),'Previstos = ',len(anotacoes_previstas['boxes']))

    # Coloca os retângulos reais (anotação)
    # Precisa permutar o tensor para que a imagem fique nas primeiras duas dimensões
    imagem = cria_imagem_anotada(imagem.permute(1, 2, 0).cpu().numpy(),
                                 anotacoes,(0,0,255))    
    # Coloca os retângulos previstos pelo rede (não precisa permutar novamente)
    imagem = cria_imagem_anotada(imagem,anotacoes_previstas,(0,255,0),espessura=1)

    # Adiciona a imagem na grade que será mostrada
    figure.add_subplot(rows, cols, i+1)
    plt.imshow(imagem)
    
plt.show() # Este é o comando que vai mostrar as imagens

"""## Gerando algumas estatísticas no conjunto de teste

AINDA NÃO FOI IMPLEMENTADO
"""
