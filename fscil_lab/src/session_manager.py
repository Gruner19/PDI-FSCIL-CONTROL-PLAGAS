import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from src.dataset_loader import preparar_dados_sessao
from src.preprocessing import pipeline_completo_preprocessamento
from src.feature_extraction import extrair_todas_caracteristicas
from src.prototype_memory import MemoriaDePrototipos
from src.metrics import (
    calcular_metricas_sessao,
    gerar_relatorio_sessoes,
    montar_matriz_confusao,
)

# As exceções documentadas à regra de vetorização são:
# 1. GLCM (graycomatrix) opera imagem a imagem — limitação da biblioteca scikit-image.
# 2. O laço de sessões abaixo é conceitualmente sequencial por definição do problema FSCIL:
#    cada sessão acontece após a anterior, não havendo paralelismo possível na
#    orquestração temporal das sessões.


class GerenciadorDeSessoes:
    """Orquestra a execução do protocolo FSCIL sessão a sessão."""

    def __init__(
        self,
        dados_brutos: Dict[str, object],
        sessoes: Dict[str, object],
        tamanho_alvo: tuple = (64, 64),
        incluir_hog: bool = False,
        semente_aleatoria: int = 42,
        nomes_classes_externos: Optional[List[str]] = None,
    ):
        self.dados_brutos = dados_brutos
        self.sessoes = sessoes
        self.tamanho_alvo = tamanho_alvo
        self.incluir_hog = incluir_hog
        self.semente_aleatoria = semente_aleatoria
        self.memoria = MemoriaDePrototipos(nomes_classes=nomes_classes_externos)
        self.escalonador = None
        self.historico_acuracias: List[float] = []
        self.historico_matrizes: List[np.ndarray] = []
        self.historico_por_classe: Optional[np.ndarray] = None
        self.numero_sessoes_executadas = 0
        self.total_classes_vistas: List[int] = []

    def _extrair_caracteristicas_lote(
        self, imagens: np.ndarray
    ) -> np.ndarray:
        """Pré-processa e extrai características de um lote de imagens."""
        if len(imagens) == 0:
            return np.empty((0, 1))
        preprocessado = pipeline_completo_preprocessamento(
            imagens, tamanho_alvo=self.tamanho_alvo
        )
        caracteristicas, self.escalonador = extrair_todas_caracteristicas(
            preprocessado["imagens_cinza"],
            preprocessado["mascaras_binarias"],
            incluir_hog=self.incluir_hog,
            escalonador=self.escalonador,
        )
        return caracteristicas

    def executar_sessao_base(self, quantidade_por_classe: int = None) -> Dict[str, object]:
        """Executa a sessão base: extrai características e inicializa memória de protótipos."""
        classes_sessao = self.sessoes["sessao_base"]
        self.total_classes_vistas = list(classes_sessao)
        dados_sessao = preparar_dados_sessao(
            self.dados_brutos,
            classes_sessao,
            quantidade_por_classe=quantidade_por_classe,
            semente_aleatoria=self.semente_aleatoria,
            modo_treino=True,
        )
        caracteristicas = self._extrair_caracteristicas_lote(dados_sessao["imagens"])
        rotulos = dados_sessao["rotulos"]
        self.memoria.adicionar_classes(caracteristicas, rotulos)
        self._avaliar(rotulos, caracteristicas)
        self.numero_sessoes_executadas = 1
        return {
            "sessao": 0,
            "classes": classes_sessao,
            "metricas": self.historico_acuracias[-1],
            "numero_amostras": len(rotulos),
        }

    def executar_sessao_incremental(
        self,
        indice_sessao: int,
        quantidade_por_classe: int = 5,
        estrategia_atualizacao: str = "media_simples",
        fator_suavizacao: float = 0.5,
    ) -> Dict[str, object]:
        """Processa uma sessão incremental e avalia sobre todas as classes vistas."""
        if indice_sessao >= len(self.sessoes["sessoes_incrementais"]):
            raise ValueError(
                f"Índice de sessão {indice_sessao} inválido. "
                f"Máximo: {len(self.sessoes['sessoes_incrementais']) - 1}"
            )
        classes_novas = self.sessoes["sessoes_incrementais"][indice_sessao]
        self.total_classes_vistas.extend(classes_novas)
        dados_sessao = preparar_dados_sessao(
            self.dados_brutos,
            classes_novas,
            quantidade_por_classe=quantidade_por_classe,
            semente_aleatoria=self.semente_aleatoria + indice_sessao,
            modo_treino=True,
        )
        caracteristicas_novas = self._extrair_caracteristicas_lote(
            dados_sessao["imagens"]
        )
        rotulos_novos = dados_sessao["rotulos"]
        self.memoria.adicionar_classes(caracteristicas_novas, rotulos_novos)
        if estrategia_atualizacao == "media_simples":
            pass
        elif estrategia_atualizacao == "media_movel_exponencial":
            self.memoria.atualizar_prototipos_existentes(
                caracteristicas_novas, rotulos_novos, fator_suavizacao
            )
        elif estrategia_atualizacao == "recalibracao_por_contagem":
            self.memoria.recalibrar_prototipos_por_contagem(
                caracteristicas_novas, rotulos_novos
            )
        else:
            raise ValueError(f"Estratégia de atualização desconhecida: {estrategia_atualizacao}")
        # Avaliar em TODAS as classes vistas
        self._avaliar()
        self.numero_sessoes_executadas += 1
        return {
            "sessao": indice_sessao + 1,
            "classes_novas": classes_novas,
            "metrica": self.historico_acuracias[-1],
        }

    def _avaliar(
        self, rotulos_reais: np.ndarray = None,
        caracteristicas: np.ndarray = None
    ) -> None:
        """Avalia a memória atual em todas as classes vistas."""
        if rotulos_reais is None or caracteristicas is None:
            dados_teste = preparar_dados_sessao(
                self.dados_brutos,
                self.total_classes_vistas,
                quantidade_por_classe=None,
                semente_aleatoria=self.semente_aleatoria,
                modo_treino=False,
            )
            if len(dados_teste["imagens"]) == 0:
                self.historico_acuracias.append(0.0)
                self.historico_matrizes.append(np.empty((0, 0)))
                return
            caracteristicas = self._extrair_caracteristicas_lote(
                dados_teste["imagens"]
            )
            rotulos_reais = dados_teste["rotulos"]
        rotulos_preditos = self.memoria.classificar(caracteristicas)
        metricas = calcular_metricas_sessao(rotulos_reais, rotulos_preditos)
        self.historico_acuracias.append(metricas["acuracia"])
        classes_ordenadas = sorted(set(rotulos_reais) | set(rotulos_preditos))
        self.historico_matrizes.append(
            montar_matriz_confusao(
                rotulos_reais, rotulos_preditos, classes=classes_ordenadas
            )
        )

    def obter_relatorio(self) -> Dict[str, object]:
        """Gera o relatório completo de todas as sessões executadas até o momento."""
        return gerar_relatorio_sessoes(
            self.historico_acuracias,
            self.historico_matrizes,
            self.historico_por_classe,
        )

    def obter_numero_sessoes_disponiveis(self) -> int:
        return 1 + len(self.sessoes["sessoes_incrementais"])
