# Converte anotações do tipo polígonos em arquivo JSON
# para máscaras em arquivo PNG
#
# Antes de rodar, copie as imagens anotadas pelo labelme
# para a pasta chamada imagens. Nas imagens anotadas
# pelo labelme teremos os polígonos em arquivo (.json)
# na mesma pasta das imagens. Este script vai converter
# estas anotações para o formato de máscaras e colocá-las
# na pasta chamada "anotacoes".

mkdir -p anotacoes
cd imagens
for imagem in *.json; do
    echo 'Processando ' $imagem
    semExtensao="${imagem%.*}"
    labelme_json_to_dataset $imagem -o ../anotacoes/$imagem
    cd ../anotacoes
    mv $imagem/label.png .
    nomePNG=${semExtensao}.png
    mv label.png $nomePNG
    rm -rf ./$imagem/
    cd ../imagens
done   

# Apagar os arquivos json originais
rm *.json
cd ..
echo 'Terminei...'
   

