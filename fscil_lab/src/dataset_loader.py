import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def gerar_dataset_sintetico(
    quantidade_imagens: int = 5000,
    numero_classes: int = 50,
    altura: int = 64,
    largura: int = 64,
    semente_aleatoria: int = 42,
) -> Dict[str, object]:
    """Gera dataset sintético para testes, sem necessidade de download."""
    gerador = np.random.default_rng(semente_aleatoria)
    imagens = gerador.integers(0, 256, size=(quantidade_imagens, altura, largura, 3), dtype=np.uint8)
    for indice in range(quantidade_imagens):
        ruido = gerador.normal(128, 40, size=(altura, largura, 3)).astype(np.uint8)
        imagens[indice] = np.clip(imagens[indice] * 0.3 + ruido * 0.7, 0, 255).astype(np.uint8)
    rotulos = np.array([indice % numero_classes for indice in range(quantidade_imagens)], dtype=np.int64)
    nomes_classes = [f"Classe_Sintetica_{indice:02d}" for indice in range(numero_classes)]
    return {"imagens": imagens, "rotulos": rotulos, "nomes_classes": nomes_classes}


def carregar_dataset_bruto(
    nome_dataset: str, diretorio_dados: str
) -> Dict[str, object]:
    if nome_dataset.lower() == "cifar100":
        return _carregar_cifar100(diretorio_dados)
    elif nome_dataset.lower() == "plantvillage":
        return _carregar_plantvillage(diretorio_dados)
    elif nome_dataset.lower() == "plantdoc":
        return _carregar_plantdoc(diretorio_dados)
    elif nome_dataset.lower() == "sintetico":
        return gerar_dataset_sintetico(semente_aleatoria=42)
    else:
        raise ValueError(
            f"Dataset '{nome_dataset}' não suportado. Opções: cifar100, sintetico, plantvillage, plantdoc."
        )


def _carregar_cifar100(diretorio_dados: str) -> Dict[str, object]:
    """Carrega CIFAR-100 de arquivos locais."""
    caminho_cache = Path(diretorio_dados) / "cifar100"
    arquivo_metadados = caminho_cache / "meta"
    arquivo_treino = caminho_cache / "train"
    arquivo_teste = caminho_cache / "test"
    if not (arquivo_treino.exists() and arquivo_teste.exists() and arquivo_metadados.exists()):
        raise FileNotFoundError(
            f"Arquivos CIFAR-100 não encontrados em {caminho_cache}. "
            "Coloque os arquivos 'meta', 'train' e 'test' (extraídos do CIFAR-100) em:\n"
            f"  {caminho_cache}/\n\n"
            "Passos:\n"
            "  1. Baixe de: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz\n"
            "  2. Extraia o .tar.gz\n"
            "  3. Copie 'meta', 'train' e 'test' de cifar-100-python/ para:\n"
            f"     {caminho_cache}/\n"
        )
    with open(arquivo_treino, "rb") as arquivo:
            dados_treino = pickle.load(arquivo, encoding="bytes")
    with open(arquivo_teste, "rb") as arquivo:
            dados_teste = pickle.load(arquivo, encoding="bytes")
    with open(arquivo_metadados, "rb") as arquivo:
            dados_meta = pickle.load(arquivo, encoding="bytes")
    imagens = np.concatenate([dados_treino[b"data"], dados_teste[b"data"]], axis=0)
    imagens = imagens.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    rotulos = np.concatenate(
        [np.array(dados_treino[b"fine_labels"]), np.array(dados_teste[b"fine_labels"])],
        axis=0,
    )
    nomes_classes = [nome.decode("utf-8") for nome in dados_meta[b"fine_label_names"]]
    return {
        "imagens": imagens.astype(np.uint8),
        "rotulos": rotulos,
        "nomes_classes": nomes_classes,
    }


def _carregar_plantvillage(diretorio_dados: str) -> Dict[str, object]:
    caminho_raiz = Path(diretorio_dados) / "plantvillage" / "color"
    if not caminho_raiz.exists():
        raise FileNotFoundError(
            f"Diretório PlantVillage não encontrado em {caminho_raiz}. "
            "Baixe manualmente de:\n"
            "  https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color.zip\n"
            "e extraia para data/raw/plantvillage/ de modo que a estrutura seja:\n"
            "  data/raw/plantvillage/color/Apple___Apple_scab/\n"
            "  data/raw/plantvillage/color/Tomato___Late_blight/\n"
            "  ..."
        )
    from skimage.io import imread

    classes = sorted(os.listdir(caminho_raiz))
    mapeamento_rotulos = {nome: indice for indice, nome in enumerate(classes)}
    todas_imagens = []
    todos_rotulos = []
    for nome_classe in classes:
        caminho_classe = caminho_raiz / nome_classe
        if not caminho_classe.is_dir():
            continue
        for arquivo in sorted(os.listdir(caminho_classe)):
            caminho_arquivo = caminho_classe / arquivo
            try:
                imagem = imread(caminho_arquivo)
                if imagem.ndim == 3 and imagem.shape[2] >= 3:
                    imagem = imagem[:, :, :3]
                elif imagem.ndim == 2:
                    from skimage.color import gray2rgb
                    imagem = gray2rgb(imagem)
                todas_imagens.append(imagem)
                todos_rotulos.append(mapeamento_rotulos[nome_classe])
            except Exception:
                continue
    if not todas_imagens:
        raise ValueError(
            f"Nenhuma imagem encontrada em {caminho_raiz}. "
            "Verifique se o dataset PlantVillage foi baixado corretamente "
            "e se os diretórios de classe contêm imagens."
        )
    imagens = np.stack(todas_imagens, axis=0)
    rotulos = np.array(todos_rotulos, dtype=np.int64)
    return {
        "imagens": imagens,
        "rotulos": rotulos,
        "nomes_classes": classes,
    }


def _carregar_plantdoc(diretorio_dados: str) -> Dict[str, object]:
    caminho_raiz = Path(diretorio_dados) / "plantdoc"
    if not caminho_raiz.exists():
        raise FileNotFoundError(
            f"Diretório PlantDoc não encontrado em {caminho_raiz}. "
            "Baixe manualmente de:\n"
            "  https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip\n"
            "e extraia para data/raw/plantdoc/"
        )
    from skimage.io import imread

    # Si hay subdiretorios train/ e test/, carregar de ambos
    if (caminho_raiz / "train").is_dir() and (caminho_raiz / "test").is_dir():
        todas_imagens = []
        todos_rotulos = []
        classes = []
        mapeamento_rotulos = {}
        for subset in ("train", "test"):
            subset_path = caminho_raiz / subset
            for nome_classe in sorted(os.listdir(subset_path)):
                if nome_classe not in mapeamento_rotulos:
                    mapeamento_rotulos[nome_classe] = len(classes)
                    classes.append(nome_classe)
                caminho_classe = subset_path / nome_classe
                if not caminho_classe.is_dir():
                    continue
                for arquivo in sorted(os.listdir(caminho_classe)):
                    try:
                        imagem = imread(caminho_classe / arquivo)
                        if imagem.ndim == 3 and imagem.shape[2] >= 3:
                            imagem = imagem[:, :, :3]
                        elif imagem.ndim == 2:
                            from skimage.color import gray2rgb
                            imagem = gray2rgb(imagem)
                        todas_imagens.append(imagem)
                        todos_rotulos.append(mapeamento_rotulos[nome_classe])
                    except Exception:
                        continue
        if not todas_imagens:
            raise ValueError(
                f"Nenhuma imagem encontrada em {caminho_raiz}/train ou /test. "
                "Verifique se o dataset PlantDoc foi baixado corretamente."
            )
        imagens = np.stack(todas_imagens, axis=0)
        rotulos = np.array(todos_rotulos, dtype=np.int64)
        return {"imagens": imagens, "rotulos": rotulos, "nomes_classes": classes}

    # Fallback: estrutura plana (diretórios de classe na raiz)
    classes = sorted(os.listdir(caminho_raiz))
    mapeamento_rotulos = {nome: indice for indice, nome in enumerate(classes)}
    todas_imagens = []
    todos_rotulos = []
    for nome_classe in classes:
        caminho_classe = caminho_raiz / nome_classe
        if not caminho_classe.is_dir():
            continue
        for arquivo in sorted(os.listdir(caminho_classe)):
            caminho_arquivo = caminho_classe / arquivo
            try:
                imagem = imread(caminho_arquivo)
                if imagem.ndim == 3 and imagem.shape[2] >= 3:
                    imagem = imagem[:, :, :3]
                elif imagem.ndim == 2:
                    from skimage.color import gray2rgb
                    imagem = gray2rgb(imagem)
                todas_imagens.append(imagem)
                todos_rotulos.append(mapeamento_rotulos[nome_classe])
            except Exception:
                continue
    if not todas_imagens:
        raise ValueError(
            f"Nenhuma imagem encontrada em {caminho_raiz}. "
            "Verifique se o dataset PlantDoc foi baixado corretamente "
            "e se os diretórios de classe contêm imagens."
        )
    imagens = np.stack(todas_imagens, axis=0)
    rotulos = np.array(todos_rotulos, dtype=np.int64)
    return {
        "imagens": imagens,
        "rotulos": rotulos,
        "nomes_classes": classes,
    }


def dividir_em_sessoes(
    rotulos: np.ndarray,
    numero_classes_base: int,
    classes_por_sessao_incremental: int,
    semente_aleatoria: int,
    nomes_classes_prioritarias: Optional[List[str]] = None,
) -> Dict[str, object]:
    classes_unicas = np.unique(rotulos)
    gerador = np.random.default_rng(semente_aleatoria)
    if nomes_classes_prioritarias is not None:
        raise NotImplementedError(
            "Seleção prioritária de classes na sessão base não implementada."
        )
    classes_embaralhadas = classes_unicas.copy()
    gerador.shuffle(classes_embaralhadas)
    sessao_base = classes_embaralhadas[:numero_classes_base].tolist()
    classes_restantes = classes_embaralhadas[numero_classes_base:]
    sessoes_incrementais = []
    for indice in range(0, len(classes_restantes), classes_por_sessao_incremental):
        sessao = classes_restantes[
            indice : indice + classes_por_sessao_incremental
        ].tolist()
        sessoes_incrementais.append(sessao)
    return {
        "sessao_base": sessao_base,
        "sessoes_incrementais": sessoes_incrementais,
    }


def amostrar_poucos_exemplos(
    imagens_classe: np.ndarray, quantidade_por_classe: int, semente_aleatoria: int
) -> np.ndarray:
    gerador = np.random.default_rng(semente_aleatoria)
    indices = gerador.choice(
        imagens_classe.shape[0],
        size=min(quantidade_por_classe, imagens_classe.shape[0]),
        replace=False,
    )
    return imagens_classe[indices]


def preparar_dados_sessao(
    dados_brutos: Dict[str, object],
    classes_sessao: List[int],
    quantidade_por_classe: Optional[int] = None,
    semente_aleatoria: int = 42,
    modo_treino: bool = True,
) -> Dict[str, object]:
    """Prepara os dados de uma sessão específica.

    Args:
        dados_brutos: Dicionário com 'imagens', 'rotulos', 'nomes_classes'.
        classes_sessao: Lista de índices de classes incluídas nesta sessão.
        quantidade_por_classe: Se fornecido, amostra K exemplos por classe.
        semente_aleatoria: Semente para reprodutibilidade.
        modo_treino: Se True, aplica amostragem quando quantidade_por_classe é fornecido.

    Returns:
        Dicionário com 'imagens', 'rotulos' para esta sessão.
    """
    imagens = dados_brutos["imagens"]
    rotulos = dados_brutos["rotulos"]
    mascara = np.isin(rotulos, classes_sessao)
    imagens_sessao = imagens[mascara]
    rotulos_sessao = rotulos[mascara]
    if modo_treino and quantidade_por_classe is not None:
        imagens_amostradas = []
        rotulos_amostrados = []
        for classe in classes_sessao:
            mascara_classe = rotulos_sessao == classe
            imagens_classe = imagens_sessao[mascara_classe]
            if len(imagens_classe) == 0:
                continue
            amostra = amostrar_poucos_exemplos(
                imagens_classe, quantidade_por_classe, semente_aleatoria
            )
            imagens_amostradas.append(amostra)
            rotulos_amostrados.append(np.full(amostra.shape[0], classe, dtype=np.int64))
        if len(imagens_amostradas) == 0:
            return {"imagens": np.empty((0, *imagens.shape[1:]), dtype=imagens.dtype),
                    "rotulos": np.empty((0,), dtype=np.int64)}
        imagens_sessao = np.concatenate(imagens_amostradas, axis=0)
        rotulos_sessao = np.concatenate(rotulos_amostrados, axis=0)
    return {"imagens": imagens_sessao, "rotulos": rotulos_sessao}
