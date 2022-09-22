# Exemplos de códigos para redes neurais profundas com pytorch

__autor__: Hemerson Pistori (pistori@ucdb.br)
	   Adaptado dos tutoriais do pytorch


Exemplos documentados em ordem crescente de complexidade de
códigos para trabalhar com redes neurais profundas. 

### Exemplo de uso no colab:

  Obs: cada versão abaixo tem seu próprio ambiente de execução, recomenda-se criar uma cópia no seu drive e executar por ela para poder salvar quaisquer alterações que você julgue necessário.

Links para o colab de cada arquivo a seguir:

- [exemplo_pytorch_v1.py](https://colab.research.google.com/drive/1sJJgfc_2wLvvZWwhz2Ea8oWUxS9IcORu) Classificação com o FashionMNIST
- [exemplo_pytorch_v2.py](https://colab.research.google.com/drive/1eqZbgFoN2GLNFBreSx1Rp7DmU6b3Di7E) Classificação com o FashionMNIST usando parada antecipada, conjunto de teste nosso, métricas de desempenho e matriz de confusão
- [exemplo_pytorch_v3.py](https://colab.research.google.com/drive/1-GMeHTbbz4MqqUDOMPkd2cLT0rMIBU8k) Classificação com o FashionMNIST usando redes prontas e transfer learning
- [exemplo_pytorch_v4.py](https://colab.research.google.com/drive/1egrQOlXvi_rvX2ZtfvK56wVXMyIp6GCh) Classificação com um banco de imagens próprio
- [exemplo_pytorch_v5.py](https://colab.research.google.com/drive/1XXegdU79g7HuNvtlSkDzrcbdL67Y-g9q) Segmentação semântica 
- [exemplo_pytorch_v6.py](https://colab.research.google.com/drive/1YNMPsOhR2PV-DDexVmgo8Fb6mVbTMga6) Detecção de objetos

- [coisas_bacanas_de_python.py](https://colab.research.google.com/drive/1nET98zFfKdDXXWYofQyPZ-uavYTd0y8Q) Seleção de exemplos de coisas bacanas em python que não existem ou não são tão comuns em linguagens como Java e C

### Exemplo de uso em máquina local:

  Obs: Verifique qual versão do cudatoolkit é compatível com a sua e troque se necessário. Se a máquina não tiver GPU, NÃO tente instalar o cudatoolkit (retire ele do conda create)

```
conda create -n pytorch -c conda-forge matplotlib pytorch torchvision tensorboard scikit-learn pandas pillow seaborn cudatoolkit=10.1
conda activate pytorch
python exemplo_pytorch_v1.py
```

Para rodar o segundo exemplo e depois analisar resultados com o tensorboard:

```
conda activate pytorch
python exemplo_pytorch_v2.py
tensorboard serve --logdir ./runs/
Abrir em um navegador este link aqui: http://localhost:6006/
```

