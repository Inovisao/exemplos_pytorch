# Testa se o torch está instalado e se está encontrando a GPU !!!

import torch   # Pytorch principal

# Verifica se tem GPU na máquina, caso contrário, usa a CPU mesmo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {device}")


