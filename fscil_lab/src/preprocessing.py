import numpy as np
from skimage.transform import resize
from skimage.filters import threshold_sauvola
from skimage.color import rgb2gray


def converter_para_escala_cinza(lote_imagens: np.ndarray) -> np.ndarray:
    """Conversão vetorizada RGB->cinza para um lote de imagens (N, H, W, 3) -> (N, H, W)."""
    if lote_imagens.ndim == 3:
        return lote_imagens
    if lote_imagens.shape[-1] == 1:
        return lote_imagens.squeeze(-1)
    return np.array([rgb2gray(imagem) for imagem in lote_imagens])


def redimensionar_lote(
    lote_imagens: np.ndarray, tamanho_alvo: tuple
) -> np.ndarray:
    """Redimensiona um lote de imagens para tamanho_alvo (H, W)."""
    if lote_imagens.ndim == 3:
        return np.array(
            [resize(imagem, tamanho_alvo, preserve_range=True).astype(np.uint8)
             for imagem in lote_imagens]
        )
    return np.array(
        [
            resize(imagem, tamanho_alvo, preserve_range=True).astype(np.uint8)
            for imagem in lote_imagens
        ]
    )


def normalizar_intensidade(lote_imagens: np.ndarray) -> np.ndarray:
    """Normalização min-max vetorizada por imagem, usando broadcasting."""
    minimos = lote_imagens.min(axis=(1, 2), keepdims=True)
    maximos = lote_imagens.max(axis=(1, 2), keepdims=True)
    alcance = maximos - minimos
    alcance = np.where(alcance == 0, 1, alcance)
    return (lote_imagens.astype(np.float64) - minimos) / alcance


def binarizar_sauvola_lote(
    lote_imagens_cinza: np.ndarray, tamanho_janela: int = 25
) -> np.ndarray:
    """Aplica limiarização adaptativa de Sauvola retornando máscaras binárias (N, H, W)."""
    if lote_imagens_cinza.dtype != np.float64:
        lote_imagens_cinza = lote_imagens_cinza.astype(np.float64)
    if lote_imagens_cinza.max() > 1.0:
        lote_imagens_cinza = lote_imagens_cinza / 255.0
    return np.array(
        [
            threshold_sauvola(imagem, window_size=tamanho_janela)
            for imagem in lote_imagens_cinza
        ]
    )


def pipeline_completo_preprocessamento(
    lote_imagens: np.ndarray, tamanho_alvo: tuple = (64, 64)
) -> dict:
    """Aplica todo o pipeline de pré-processamento.

    Returns:
        dict com 'imagens_cinza', 'imagens_normalizadas', 'mascaras_binarias'.
    """
    if lote_imagens.ndim == 4 and lote_imagens.shape[-1] == 3:
        imagens_redimensionadas = redimensionar_lote(lote_imagens, tamanho_alvo)
        imagens_cinza = converter_para_escala_cinza(imagens_redimensionadas)
    else:
        imagens_redimensionadas = redimensionar_lote(lote_imagens, tamanho_alvo)
        imagens_cinza = imagens_redimensionadas
    imagens_normalizadas = normalizar_intensidade(imagens_cinza)
    mascaras_binarias = binarizar_sauvola_lote(imagens_cinza)
    return {
        "imagens_cinza": imagens_cinza,
        "imagens_normalizadas": imagens_normalizadas,
        "mascaras_binarias": mascaras_binarias.astype(np.float64),
    }
