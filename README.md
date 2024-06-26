# Exemplos de códigos para redes neurais profundas com pytorch

__autor__: Hemerson Pistori (pistori@ucdb.br)
	   Adaptado dos tutoriais do pytorch


Exemplos documentados em ordem crescente de complexidade de
códigos para trabalhar com redes neurais profundas. 

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

### Dicas de uso no colab:

- Crie uma conta no [Google colab](https://colab.research.google.com/)
- Dentro do Colab crie um novo notebook jupyter
- Copie e cole o conteúdo de algum dos exemplos para dentro do notebook
- Siga as orientações de dentro do código. Note que o código foi gerado a partir de um notebook jupyter e tem algumas coisas documentadas que podem ter que ser "descomentadas"
- Pode ser interessante quebrar o código em várias caixas diferentes no colab para poder fazer experimentos com pequenos trechos

### Coisas Bacanas em Python

- Link direto para o exemplo das coisas bacanas em python no colab:[CLIQUE AQUI](https://colab.research.google.com/drive/1mwdCGpZTomGO3VnKVaHkORwDY0o5yygJ?usp=sharing)
  


  
  
