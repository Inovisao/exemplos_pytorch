# Exemplos de códigos para redes neurais profundas com pytorch

__autor__: Hemerson Pistori (pistori@ucdb.br)
	   Adaptado dos tutoriais do pytorch


Exemplos documentados em ordem crescente de complexidade de
códigos para trabalhar com redes neurais profundas. 

### Coisas Bacanas em Python

- Link direto para o exemplo das coisas bacanas em python no colab:[CLIQUE AQUI](https://colab.research.google.com/drive/1mwdCGpZTomGO3VnKVaHkORwDY0o5yygJ?usp=sharing)

### Dicas para rodar os exemplos no Google Colab:

- Crie uma conta no [Google colab](https://colab.research.google.com/)
- Dentro do Colab crie um novo notebook jupyter
- Copie e cole o conteúdo de algum dos exemplos para dentro do notebook (em um campo de código - code)
- Altere o ambiente de execução (runtime type) para usar uma GPU e demorar menos para rodar
- Clique no botão de play no lado esquerdo da caixa de código


### Exemplo de uso em máquina local:


```
conda create -n pytorch -c conda-forge matplotlib pytorch torchvision tensorboard scikit-learn pandas pillow seaborn cudatoolkit
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

  


  
  
