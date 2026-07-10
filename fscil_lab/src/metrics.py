import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from typing import Dict, List, Optional


def calcular_acuracia(
    rotulos_verdadeiros: np.ndarray, rotulos_preditos: np.ndarray
) -> float:
    """Acurácia vetorizada."""
    if len(rotulos_verdadeiros) == 0:
        return 0.0
    return float(np.mean(rotulos_verdadeiros == rotulos_preditos))


def calcular_performance_dropping_rate(
    acuracia_sessao_base: float, acuracia_ultima_sessao: float
) -> float:
    """PD = acuracia_sessao_base - acuracia_ultima_sessao."""
    return acuracia_sessao_base - acuracia_ultima_sessao


def calcular_esquecimento_medio(
    historico_acuracias_por_classe: np.ndarray,
) -> float:
    """Diferença entre o pico histórico e o valor final de acurácia por classe, vetorizada.

    Args:
        historico_acuracias_por_classe: Matriz (numero_classes, numero_sessoes)
            com a acurácia de cada classe em cada sessão.

    Returns:
        Esquecimento médio (float).
    """
    if historico_acuracias_por_classe.shape[0] == 0:
        return 0.0
    picos_historicos = np.max(historico_acuracias_por_classe, axis=1)
    valores_finais = historico_acuracias_por_classe[:, -1]
    esquecimentos = picos_historicos - valores_finais
    return float(np.mean(esquecimentos))


def montar_matriz_confusao(
    rotulos_verdadeiros: np.ndarray,
    rotulos_preditos: np.ndarray,
    classes: Optional[List[int]] = None,
) -> np.ndarray:
    """Matriz de confusão via sklearn.metrics.confusion_matrix."""
    return sk_confusion_matrix(
        rotulos_verdadeiros, rotulos_preditos, labels=classes
    )


def calcular_metricas_sessao(
    rotulos_verdadeiros: np.ndarray,
    rotulos_preditos: np.ndarray,
) -> Dict[str, object]:
    """Calcula todas as métricas para uma sessão."""
    return {
        "acuracia": calcular_acuracia(rotulos_verdadeiros, rotulos_preditos),
        "matriz_confusao": montar_matriz_confusao(
            rotulos_verdadeiros, rotulos_preditos
        ),
        "total_amostras": len(rotulos_verdadeiros),
        "acertos": int(np.sum(rotulos_verdadeiros == rotulos_preditos)),
    }


def gerar_relatorio_sessoes(
    historico_acuracias: List[float],
    historico_matrizes: List[np.ndarray],
    historico_por_classe: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Gera relatório completo com PD e forgetting."""
    acuracia_media = float(np.mean(historico_acuracias))
    pd = calcular_performance_dropping_rate(
        historico_acuracias[0], historico_acuracias[-1]
    )
    forgetting = 0.0
    if historico_por_classe is not None:
        forgetting = calcular_esquecimento_medio(historico_por_classe)
    return {
        "historico_acuracias": historico_acuracias,
        "acuracia_media": acuracia_media,
        "performance_dropping_rate": pd,
        "esquecimento_medio": forgetting,
        "historico_matrizes": historico_matrizes,
    }
