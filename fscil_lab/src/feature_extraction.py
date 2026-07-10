import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.measure import moments_hu, regionprops_table
from skimage.morphology import label
from sklearn.preprocessing import StandardScaler

# Exceções documentadas à regra de vetorização (seção 7.2 do projeto):
# 1. graycomatrix — não possui API vetorizada em lote; opera imagem a imagem.
# 2. moments_hu — idem; processa uma imagem 2D por chamada.
# 3. regionprops_table — idem; requer imagem rotulada individual.
# 4. hog — idem; skimage.feature.hog não é batch-vectorized.
# Todas são limitações legítimas das bibliotecas subjacentes,
# não escolhas de implementação.


def extrair_descritores_glcm(
    lote_imagens_cinza: np.ndarray,
    distancias: list = None,
    angulos: list = None,
) -> np.ndarray:
    """Retorna matriz (N, D_glcm) com contraste, homogeneidade, energia, correlação, ASM.

    Agregado por média entre distâncias/ângulos para cada propriedade.
    """
    if distancias is None:
        distancias = [1, 3, 5]
    if angulos is None:
        angulos = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    if lote_imagens_cinza.max() <= 1.0:
        lote_imagens_cinza = (lote_imagens_cinza * 255).astype(np.uint8)
    propriedades = ["contrast", "homogeneity", "energy", "correlation", "ASM"]
    resultados = []
    for imagem in lote_imagens_cinza:
        imagem_uint8 = imagem.astype(np.uint8)
        glcm = graycomatrix(
            imagem_uint8, distances=distancias, angles=angulos,
            levels=256, symmetric=True, normed=True
        )
        vetor_imagem = []
        for propriedade in propriedades:
            valores = graycoprops(glcm, prop=propriedade)
            vetor_imagem.append(valores.mean())
        resultados.append(vetor_imagem)
    return np.array(resultados)


def extrair_momentos_hu(lote_imagens_cinza: np.ndarray) -> np.ndarray:
    """Retorna matriz (N, 7) de momentos de Hu (log-transformados para estabilidade numérica)."""
    if lote_imagens_cinza.max() > 1.0:
        lote_imagens_cinza = lote_imagens_cinza / 255.0
    resultados = []
    for imagem in lote_imagens_cinza:
        momentos = moments_hu(imagem.astype(np.float64))
        momentos_log = -np.sign(momentos) * np.log(np.abs(momentos) + 1e-10)
        resultados.append(momentos_log)
    return np.array(resultados)


def extrair_descritores_hog(
    lote_imagens_cinza: np.ndarray,
    orientacoes: int = 9,
    pixels_por_celula: int = 8,
    celulas_por_bloco: int = 2,
) -> np.ndarray:
    """Retorna matriz (N, D_hog) com descritores HOG."""
    resultados = []
    for imagem in lote_imagens_cinza:
        vetor_hog = hog(
            imagem.astype(np.float64),
            orientations=orientacoes,
            pixels_per_cell=(pixels_por_celula, pixels_por_celula),
            cells_per_block=(celulas_por_bloco, celulas_por_bloco),
            feature_vector=True,
        )
        resultados.append(vetor_hog)
    return np.array(resultados)


def extrair_descritores_morfologicos(
    lote_mascaras_binarias: np.ndarray
) -> np.ndarray:
    """Retorna matriz (N, D_morf) com área, perímetro, excentricidade, solidez via regionprops.

    Para imagens com múltiplos objetos, seleciona o maior objeto (região de interesse).
    """
    propriedades = ["area", "perimeter", "eccentricity", "solidity"]
    resultados = []
    for mascara in lote_mascaras_binarias:
        mascara_bool = mascara.astype(bool)
        if mascara_bool.sum() == 0:
            resultados.append([0.0] * len(propriedades))
            continue
        imagem_rotulada = label(mascara_bool)
        tabela = regionprops_table(
            imagem_rotulada, properties=propriedades
        )
        if len(tabela[propriedades[0]]) == 0:
            resultados.append([0.0] * len(propriedades))
            continue
        # Selecionar o maior objeto
        indices_maior = np.argmax(tabela["area"])
        vetor = [tabela[prop][indices_maior] for prop in propriedades]
        resultados.append(vetor)
    return np.array(resultados, dtype=np.float64)


def montar_vetor_caracteristicas(
    descritores_glcm: np.ndarray,
    momentos_hu: np.ndarray,
    descritores_morfologicos: np.ndarray,
    descritores_hog: np.ndarray = None,
    escalonador: StandardScaler = None,
    aplicar_escalonamento: bool = True,
) -> tuple:
    """Concatena e padroniza (StandardScaler) os descritores em um único vetor por imagem.

    Returns:
        (vetor_caracteristicas, escalonador) — o escalonador é retornado para uso futuro.
    """
    componentes = [descritores_glcm, momentos_hu, descritores_morfologicos]
    if descritores_hog is not None:
        componentes.append(descritores_hog)
    vetor = np.concatenate(componentes, axis=1)
    vetor = np.nan_to_num(vetor, nan=0.0, posinf=0.0, neginf=0.0)
    if aplicar_escalonamento:
        if escalonador is None:
            escalonador = StandardScaler()
            vetor_escalonado = escalonador.fit_transform(vetor)
        else:
            vetor_escalonado = escalonador.transform(vetor)
        return vetor_escalonado, escalonador
    return vetor, escalonador


def extrair_todas_caracteristicas(
    lote_imagens_cinza: np.ndarray,
    lote_mascaras_binarias: np.ndarray,
    incluir_hog: bool = False,
    escalonador: StandardScaler = None,
) -> tuple:
    """Atalho para extrair todas as features de uma vez."""
    descritores_glcm = extrair_descritores_glcm(lote_imagens_cinza)
    momentos_hu = extrair_momentos_hu(lote_imagens_cinza)
    descritores_morfologicos = extrair_descritores_morfologicos(lote_mascaras_binarias)
    descritores_hog = None
    if incluir_hog:
        descritores_hog = extrair_descritores_hog(lote_imagens_cinza)
    return montar_vetor_caracteristicas(
        descritores_glcm,
        momentos_hu,
        descritores_morfologicos,
        descritores_hog,
        escalonador,
    )
