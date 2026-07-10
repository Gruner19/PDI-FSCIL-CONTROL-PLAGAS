import numpy as np
import sys
import os
from pathlib import Path

caminho_raiz = Path(__file__).parent.parent
sys.path.insert(0, str(caminho_raiz))

from src.dataset_loader import (
    carregar_dataset_bruto,
    dividir_em_sessoes,
    amostrar_poucos_exemplos,
    preparar_dados_sessao,
)
from src.preprocessing import (
    converter_para_escala_cinza,
    redimensionar_lote,
    normalizar_intensidade,
    binarizar_sauvola_lote,
    pipeline_completo_preprocessamento,
)
from src.feature_extraction import (
    extrair_descritores_glcm,
    extrair_momentos_hu,
    extrair_descritores_morfologicos,
    extrair_descritores_hog,
    montar_vetor_caracteristicas,
    extrair_todas_caracteristicas,
)
from src.prototype_memory import MemoriaDePrototipos
from src.metrics import (
    calcular_acuracia,
    calcular_performance_dropping_rate,
    calcular_esquecimento_medio,
    montar_matriz_confusao,
    calcular_metricas_sessao,
    gerar_relatorio_sessoes,
)


def gerar_imagens_sinteticas(
    quantidade: int, altura: int = 32, largura: int = 32, canais: int = 3
) -> np.ndarray:
    """Gera imagens sintéticas para teste do pipeline."""
    return np.random.randint(0, 256, size=(quantidade, altura, largura, canais), dtype=np.uint8)


def test_preprocessamento():
    imagens = gerar_imagens_sinteticas(10, 64, 64)
    resultado = pipeline_completo_preprocessamento(imagens, tamanho_alvo=(32, 32))
    assert "imagens_cinza" in resultado
    assert "imagens_normalizadas" in resultado
    assert "mascaras_binarias" in resultado
    assert resultado["imagens_cinza"].shape == (10, 32, 32)
    assert resultado["imagens_normalizadas"].shape == (10, 32, 32)
    assert resultado["mascaras_binarias"].shape == (10, 32, 32)
    assert resultado["imagens_normalizadas"].min() >= 0.0
    assert resultado["imagens_normalizadas"].max() <= 1.0
    print("test_preprocessamento PASSED")


def test_extracao_glcm():
    imagens = np.random.randint(0, 256, size=(5, 32, 32), dtype=np.uint8)
    glcm = extrair_descritores_glcm(imagens)
    assert glcm.shape[0] == 5
    assert glcm.shape[1] == 5  # 5 propriedades GLCM
    assert not np.any(np.isnan(glcm))
    print("test_extracao_glcm PASSED")


def test_momentos_hu():
    imagens = np.random.rand(5, 32, 32).astype(np.float64)
    momentos = extrair_momentos_hu(imagens)
    assert momentos.shape == (5, 7)
    assert not np.any(np.isnan(momentos))
    assert not np.any(np.isinf(momentos))
    print("test_momentos_hu PASSED")


def test_descritores_morfologicos():
    mascaras = np.random.randint(0, 2, size=(5, 32, 32)).astype(np.float64)
    morf = extrair_descritores_morfologicos(mascaras)
    assert morf.shape == (5, 4)  # area, perimeter, eccentricity, solidity
    print("test_descritores_morfologicos PASSED")


def test_vetor_caracteristicas():
    glcm = np.random.rand(10, 5)
    hu = np.random.rand(10, 7)
    morf = np.random.rand(10, 4)
    vetor, _ = montar_vetor_caracteristicas(glcm, hu, morf, aplicar_escalonamento=True)
    assert vetor.shape == (10, 16)
    # Média ~0 e desvio ~1 após StandardScaler
    assert np.abs(vetor.mean()) < 1e-10
    assert np.abs(vetor.std(axis=0).mean() - 1.0) < 1e-6
    print("test_vetor_caracteristicas PASSED")


def test_memoria_prototipos():
    memoria = MemoriaDePrototipos()
    caracteristicas = np.random.rand(20, 16)
    rotulos = np.repeat([0, 1, 2, 3], 5)
    memoria.adicionar_classes(caracteristicas, rotulos)
    assert memoria.prototipos.shape == (4, 16)
    # Classificar as mesmas amostras
    predicoes = memoria.classificar(caracteristicas)
    assert len(predicoes) == 20
    acuracia = calcular_acuracia(rotulos, predicoes)
    assert acuracia > 0.0
    print("test_memoria_prototipos PASSED")


def test_memoria_atualizacao_incremental():
    memoria = MemoriaDePrototipos()
    carac_base = np.random.rand(20, 16)
    rot_base = np.repeat([0, 1], 10)
    memoria.adicionar_classes(carac_base, rot_base)
    assert memoria.prototipos.shape == (2, 16)
    carac_novas = np.random.rand(6, 16)
    rot_novas = np.repeat([2], 6)
    memoria.adicionar_classes(carac_novas, rot_novas)
    assert memoria.prototipos.shape == (3, 16)
    predicoes = memoria.classificar(carac_base)
    assert len(predicoes) == 20
    print("test_memoria_atualizacao_incremental PASSED")


def test_metricas():
    verdadeiros = np.array([0, 1, 2, 0, 1, 2])
    preditos = np.array([0, 1, 2, 0, 0, 1])
    acuracia = calcular_acuracia(verdadeiros, preditos)
    assert 0.0 <= acuracia <= 1.0
    pd = calcular_performance_dropping_rate(0.8, 0.6)
    assert abs(pd - 0.2) < 1e-10
    historico = np.array([[0.8, 0.7, 0.6], [0.9, 0.8, 0.8]])
    forgetting = calcular_esquecimento_medio(historico)
    assert forgetting >= 0.0
    matriz = montar_matriz_confusao(verdadeiros, preditos, classes=[0, 1, 2])
    assert matriz.shape == (3, 3)
    metricas = calcular_metricas_sessao(verdadeiros, preditos)
    assert "acuracia" in metricas
    relatorio = gerar_relatorio_sessoes([0.8, 0.7, 0.6], [matriz], historico)
    assert "acuracia_media" in relatorio
    assert "performance_dropping_rate" in relatorio
    print("test_metricas PASSED")


def test_dividir_sessoes():
    rotulos = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    sessoes = dividir_em_sessoes(rotulos, 5, 2, 42)
    assert len(sessoes["sessao_base"]) == 5
    total_inc = sum(len(sessao) for sessao in sessoes["sessoes_incrementais"])
    assert total_inc == 5  # classes 5..9
    assert all(len(sessao) <= 2 for sessao in sessoes["sessoes_incrementais"])
    print("test_dividir_sessoes PASSED")


def test_preparar_dados_sessao():
    imagens = np.random.randint(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
    rotulos = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    dados = {"imagens": imagens, "rotulos": rotulos, "nomes_classes": ["a", "b", "c", "d", "e"]}
    resultado = preparar_dados_sessao(
        dados, [0, 1], quantidade_por_classe=2, semente_aleatoria=42, modo_treino=True
    )
    assert resultado["imagens"].shape[0] == 4
    classes_presentes = set(np.unique(resultado["rotulos"]))
    assert classes_presentes == {0, 1}
    print("test_preparar_dados_sessao PASSED")


if __name__ == "__main__":
    test_preprocessamento()
    test_extracao_glcm()
    test_momentos_hu()
    test_descritores_morfologicos()
    test_vetor_caracteristicas()
    test_memoria_prototipos()
    test_memoria_atualizacao_incremental()
    test_metricas()
    test_dividir_sessoes()
    test_preparar_dados_sessao()
    print("\n=== TODOS OS TESTES PASSARAM ===")
