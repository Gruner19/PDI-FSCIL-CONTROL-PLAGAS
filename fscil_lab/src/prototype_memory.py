import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple


class MemoriaDePrototipos:
    """Mantém a matriz de protótipos (classes x dimensões) e o mapeamento classe->índice.

    Os protótipos são centróides das classes, calculados de forma agregada.
    A classificação é feita por distância euclidiana (NCM - Nearest Class Mean).
    """

    def __init__(self, nomes_classes: Optional[List[str]] = None):
        self.prototipos: Optional[np.ndarray] = None
        self.mapeamento_classe_para_indice: Dict[int, int] = {}
        self.mapeamento_indice_para_classe: Dict[int, int] = {}
        self.nomes_classes: Dict[int, str] = {}
        self.historico_contagem_amostras: Dict[int, int] = {}
        if nomes_classes is not None:
            for indice, nome in enumerate(nomes_classes):
                self.nomes_classes[indice] = nome

    @property
    def numero_classes(self) -> int:
        if self.prototipos is None:
            return 0
        return self.prototipos.shape[0]

    def adicionar_classes(
        self,
        caracteristicas_novas_classes: np.ndarray,
        rotulos_novas_classes: np.ndarray,
    ) -> None:
        """Calcula os centróides das novas classes (agregação vetorizada) e os adiciona à memória."""
        n_cols = caracteristicas_novas_classes.shape[1]
        dados = pd.DataFrame(
            {
                "rotulo": rotulos_novas_classes,
                **{f"f_{i}": caracteristicas_novas_classes[:, i] for i in range(n_cols)},
            }
        )
        colunas_features = [f"f_{i}" for i in range(n_cols)]
        grupos = dados.groupby("rotulo")
        centroides = grupos[colunas_features].mean().values
        rotulos_novos = grupos.first().index.values
        if self.prototipos is None:
            self.prototipos = centroides
        else:
            self.prototipos = np.concatenate([self.prototipos, centroides], axis=0)
        for rotulo in rotulos_novos:
            if rotulo not in self.mapeamento_classe_para_indice:
                indice = len(self.mapeamento_classe_para_indice)
                self.mapeamento_classe_para_indice[rotulo] = indice
                self.mapeamento_indice_para_classe[indice] = rotulo
        for rotulo, contagem in zip(
            rotulos_novos,
            dados.groupby("rotulo").size().values,
        ):
            self.historico_contagem_amostras[rotulo] = (
                self.historico_contagem_amostras.get(rotulo, 0) + int(contagem)
            )

    def atualizar_prototipos_existentes(
        self,
        caracteristicas: np.ndarray,
        rotulos: np.ndarray,
        fator_suavizacao: float = 0.5,
    ) -> None:
        """Atualiza protótipos existentes via média móvel exponencial (vetorizada)."""
        for rotulo in np.unique(rotulos):
            if rotulo not in self.mapeamento_classe_para_indice:
                continue
            indice_prototipo = self.mapeamento_classe_para_indice[rotulo]
            mascara = rotulos == rotulo
            amostras_classe = caracteristicas[mascara]
            media_nova = amostras_classe.mean(axis=0)
            prototipo_antigo = self.prototipos[indice_prototipo]
            self.prototipos[indice_prototipo] = (
                fator_suavizacao * prototipo_antigo
                + (1.0 - fator_suavizacao) * media_nova
            )

    def recalibrar_prototipos_por_contagem(
        self,
        caracteristicas: np.ndarray,
        rotulos: np.ndarray,
    ) -> None:
        """Recalcula protótipos usando média ponderada pelo número de amostras vistas.

        Dá mais peso ao protótipo antigo se ele foi estimado com muitas amostras.
        """
        for rotulo in np.unique(rotulos):
            if rotulo not in self.mapeamento_classe_para_indice:
                continue
            indice_prototipo = self.mapeamento_classe_para_indice[rotulo]
            mascara = rotulos == rotulo
            amostras_classe = caracteristicas[mascara]
            media_nova = amostras_classe.mean(axis=0)
            contagem_antiga = self.historico_contagem_amostras.get(rotulo, 1)
            contagem_nova = amostras_classe.shape[0]
            peso_antigo = contagem_antiga / (contagem_antiga + contagem_nova)
            peso_novo = contagem_nova / (contagem_antiga + contagem_nova)
            self.prototipos[indice_prototipo] = (
                peso_antigo * self.prototipos[indice_prototipo]
                + peso_novo * media_nova
            )
            self.historico_contagem_amostras[rotulo] = contagem_antiga + contagem_nova

    def classificar(
        self, caracteristicas_consulta: np.ndarray
    ) -> np.ndarray:
        """Retorna os rótulos preditos usando distância euclidiana vetorizada (cdist)."""
        distancias = cdist(caracteristicas_consulta, self.prototipos, metric="euclidean")
        indices_preditos = np.argmin(distancias, axis=1)
        return np.array([
            self.mapeamento_indice_para_classe[indice]
            for indice in indices_preditos
        ])

    def obter_distancias(
        self, caracteristicas_consulta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna distâncias a todos os protótipos e os índices dos protótipos."""
        distancias = cdist(caracteristicas_consulta, self.prototipos, metric="euclidean")
        return distancias, np.argsort(distancias, axis=1)

    def obter_mapeamento_classes_para_nomes(self) -> Dict[int, str]:
        return self.nomes_classes

    def definir_nome_classe(self, rotulo: int, nome: str) -> None:
        self.nomes_classes[rotulo] = nome

    def obter_prototipos(self) -> Optional[np.ndarray]:
        return self.prototipos

    def obter_rotulos_ordenados(self) -> List[int]:
        return [
            self.mapeamento_indice_para_classe[indice]
            for indice in range(len(self.mapeamento_indice_para_classe))
        ]
