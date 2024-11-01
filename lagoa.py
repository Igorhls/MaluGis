import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcular_profundidade(imagem):
    # Carrega a imagem em formato RGB
    img = cv2.imread(imagem)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Separa as bandas: Azul (B), Verde (G), e Vermelho (R)
    azul, verde, _ = cv2.split(img)

    # Calcula a razão entre Azul e Verde (proxy para profundidade)
    with np.errstate(divide='ignore', invalid='ignore'):
        razao_azul_verde = np.where(verde != 0, azul / verde, 0)

    # Normaliza os valores para a faixa 0-1
    profundidade_normalizada = cv2.normalize(
        razao_azul_verde, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    # Escala a profundidade estimada (0 a 10 metros, por exemplo)
    profundidade_estimativa = profundidade_normalizada * 10  # Alterar para faixa desejada

    return profundidade_estimativa

def exibir_profundidade(imagem, profundidade):
    plt.figure(figsize=(10, 5))

    # Exibe a imagem original
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(imagem), cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.axis('off')

    # Exibe a profundidade estimada
    plt.subplot(1, 2, 2)
    plt.imshow(profundidade, cmap='Blues')
    plt.colorbar(label='Profundidade (m)')
    plt.title('Profundidade Estimada')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # Caminho da imagem a ser processada
    caminho_imagem = "teste2.png"  # Substituir pelo caminho correto

    # Calcula a profundidade e exibe os resultados
    profundidade = calcular_profundidade(caminho_imagem)
    exibir_profundidade(caminho_imagem, profundidade)
