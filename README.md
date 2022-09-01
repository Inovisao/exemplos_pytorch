# Exemplos de códigos para redes neurais profundas com pytorch

__autor__: Hemerson Pistori (pistori@ucdb.br)
	   Alguns dos códigos foram adaptados de outras fontes (no próprio código eu indico as fontes que serviram como base)


Exemplos documentados em ordem crescente de complexidade de códigos para trabalhar com redes neurais profundas. Os códigos foram gerados automaticamente a partir de Notebooks Jupyter. Cada código possui um link para o Notebook Jupyter que levará para Google Colab onde pode ser executado através de um navegador (browser). Também está disponível um código como mostrar diversos recursos importantes da linguagem Python.

### Exemplo de uso:

- Obs: Verifique qual versão do cudatoolkit é compatível com a sua e troque se necessário. Se a máquina não tiver GPU, NÃO tente instalar o cudatoolkit (retire ele do conda create). Para evitar estas dificuldades no início, você pode tentar rodar primeiro no Google Colab (tem um link dentro de cada código).

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

### Dados para testes:

Alguns dos exemplos exigem dados próprios. Para estes casos (do exemplo v4 em diante), eu deixei alguns dados produzidos por nós dentro da pasta "data". São exemplos de anotações, que variam conforme o tipo de problema (classificação, segmentação semântica e detecção de objetos).

### Rodando no Google Colab como Jupyter Notebok

Links para rodar no Google Colab (faça um cópia para o seu drive se quiser
fazer alterações e salvar)

* [exemplo_pytorch_v1](https://colab.research.google.com/drive/1sJJgfc_2wLvvZWwhz2Ea8oWUxS9IcORu)
* [exemplo_pytorch_v2](https://colab.research.google.com/drive/1eqZbgFoN2GLNFBreSx1Rp7DmU6b3Di7E)
* [exemplo_pytorch_v3](https://colab.research.google.com/drive/1-GMeHTbbz4MqqUDOMPkd2cLT0rMIBU8k)
* [exemplo_pytorch_v4](https://colab.research.google.com/drive/1egrQOlXvi_rvX2ZtfvK56wVXMyIp6GCh)
* [exemplo_pytorch_v5](https://colab.research.google.com/drive/1XXegdU79g7HuNvtlSkDzrcbdL67Y-g9q)
* [exemplo_pytorch_v6](https://colab.research.google.com/drive/1YNMPsOhR2PV-DDexVmgo8Fb6mVbTMga6)


