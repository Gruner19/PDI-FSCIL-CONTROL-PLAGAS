import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.decomposition import PCA
from io import BytesIO
import os
import sys
from pathlib import Path

caminho_raiz = Path(__file__).parent
sys.path.insert(0, str(caminho_raiz))

from src.dataset_loader import (
    carregar_dataset_bruto,
    dividir_em_sessoes,
    preparar_dados_sessao as _preparar_dados_sessao,
)
from src.preprocessing import pipeline_completo_preprocessamento
from src.feature_extraction import extrair_todas_caracteristicas
from src.prototype_memory import MemoriaDePrototipos
from src.session_manager import GerenciadorDeSessoes
from src.metrics import (
    calcular_acuracia,
    montar_matriz_confusao,
    gerar_relatorio_sessoes,
)
from src.utils import dicionario_para_json
from src.pest_guide import (
    GUIA_COMPLETA,
    obtener_nombre_comun,
    obtener_cultivo,
    listar_por_cultivo,
    listar_por_tipo,
    obtener_estadisticas,
)

st.set_page_config(
    page_title="FSCIL-Lab — Diagnóstico Fitossanitário Incremental",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("FSCIL-Lab")
st.sidebar.markdown("Diagnóstico incremental de doenças foliares")
st.sidebar.markdown("---")


EMOJI_TIPO = {"enfermedad": "fungus", "plaga": "bug", "sano": "leaf"}
NOMBRE_TIPO = {"enfermedad": "Enfermedad", "plaga": "Plaga", "sano": "Sano"}


def formatear_nombre_plaga(clave) -> str:
    if not isinstance(clave, str):
        clave = str(clave)
    info = GUIA_COMPLETA.get(clave, {})
    if info:
        return f"{info['nombre_comun']} ({info['cultivo']})"
    return clave.replace("___", " - ").replace("_", " ")


def inicializar_estado():
    chaves_padrao = {
        "gerenciador": None,
        "sessao_atual": 0,
        "total_sessoes": 0,
        "executando": False,
        "configurado": False,
        "dados_brutos": None,
        "resultados": None,
        "indice_ultima_sessao": -1,
        "config_experimento": {},
    }
    for chave, valor in chaves_padrao.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def obter_nomes_dataset(dataset: str) -> list:
    exemplos = {
        "Sintético (teste rápido)": [f"Classe_Sintetica_{i:02d}" for i in range(50)],
        "CIFAR-100": [
            "apple", "aquarium_fish", "baby", "bear", "beaver",
            "bed", "bee", "beetle", "bicycle", "bottle",
            "bowl", "boy", "bridge", "bus", "butterfly",
        ],
        "PlantVillage": [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
            "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew",
            "Cherry___healthy", "Corn___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
            "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
            "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
        ],
        "PlantDoc": [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___rust",
            "Bell_pepper___Bacterial_spot", "Bell_pepper___healthy",
            "Blueberry___healthy", "Cherry___healthy", "Corn___Gray_leaf_spot",
            "Corn___Common_rust", "Corn___healthy", "Grape___Black_rot",
            "Grape___healthy", "Peach___Bacterial_spot", "Peach___healthy",
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites", "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___healthy",
        ],
    }
    return exemplos.get(dataset, [])


def plotar_curva_acuracia(acuracias: list, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    sessoes = list(range(len(acuracias)))
    ax.plot(sessoes, acuracias, marker="o", linewidth=2, markersize=8, color="#1f77b4")
    ax.fill_between(sessoes, acuracias, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Sessão", fontsize=12)
    ax.set_ylabel("Acurácia", fontsize=12)
    ax.set_title("Acurácia por Sessão", fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    for indice, valor in enumerate(acuracias):
        ax.annotate(
            f"{valor:.3f}",
            (indice, valor),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    return ax


def plotar_matriz_confusao(matriz: np.ndarray, classes: list, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    normas = matriz.astype(np.float64)
    soma_linhas = normas.sum(axis=1, keepdims=True)
    normas = np.divide(normas, soma_linhas, out=np.zeros_like(normas), where=soma_linhas > 0)
    imagem = ax.imshow(normas, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Previsto", fontsize=11)
    ax.set_ylabel("Verdadeiro", fontsize=11)
    ax.set_title("Matriz de Confusão Normalizada", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = normas[i, j]
            cor = "white" if valor > 0.5 else "black"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=6, color=cor)
    plt.colorbar(imagem, ax=ax, shrink=0.75)
    plt.tight_layout()
    return ax


def plotar_espaco_caracteristicas(
    caracteristicas: np.ndarray,
    rotulos: np.ndarray,
    prototipos: np.ndarray,
    rotulos_prototipos: list,
    nomes_classes: dict,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    pca = PCA(n_components=2, random_state=42)
    pontos_2d = pca.fit_transform(caracteristicas)
    classes_unicas = np.unique(rotulos)
    cores = plt.cm.tab20(np.linspace(0, 1, len(classes_unicas)))
    mapa_cores = {classe: cores[indice] for indice, classe in enumerate(classes_unicas)}
    for classe in classes_unicas:
        mascara = rotulos == classe
        cor = mapa_cores[classe]
        nome = nomes_classes.get(int(classe), f"Classe {classe}")
        ax.scatter(
            pontos_2d[mascara, 0],
            pontos_2d[mascara, 1],
            c=[cor],
            label=nome,
            alpha=0.5,
            s=20,
            edgecolors="none",
        )
    if prototipos is not None:
        prototipos_2d = pca.transform(prototipos)
        ax.scatter(
            prototipos_2d[:, 0],
            prototipos_2d[:, 1],
            c="red",
            marker="X",
            s=150,
            edgecolors="black",
            linewidth=1.5,
            label="Protótipos",
            zorder=5,
        )
    ax.set_xlabel("Componente Principal 1", fontsize=11)
    ax.set_ylabel("Componente Principal 2", fontsize=11)
    ax.set_title("Espaço de Características (PCA)", fontsize=13, fontweight="bold")
    ax.legend(
        fontsize=7, loc="upper left", bbox_to_anchor=(1, 1),
        markerscale=0.8, framealpha=0.8,
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Persistencia e comparacion de experimentos
# ---------------------------------------------------------------------------
import json as _json
import time as _time
DIR_EXPERIMENTOS = Path(__file__).parent / "saved_experiments"


def _garantir_dir():
    DIR_EXPERIMENTOS.mkdir(exist_ok=True)


def salvar_experimento(nome: str, config: dict, resultados: dict) -> str:
    _garantir_dir()
    timestamp = _time.strftime("%Y-%m-%d %H:%M:%S")
    dados = {
        "experiment_name": nome,
        "timestamp": timestamp,
        "config": dict(config),
        "resultados": {
            "historico_acuracias": list(resultados["historico_acuracias"]),
            "acuracia_media": float(resultados["acuracia_media"]),
            "performance_dropping_rate": float(resultados["performance_dropping_rate"]),
            "esquecimento_medio": float(resultados["esquecimento_medio"]),
            "historico_matrizes": [
                m.tolist() if hasattr(m, "tolist") else m
                for m in resultados["historico_matrizes"]
            ],
        },
    }
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in nome).strip()[:60]
    fname = f"{_time.strftime('%Y%m%d_%H%M%S')}__{safe}.json"
    path = DIR_EXPERIMENTOS / fname
    path.write_text(dicionario_para_json(dados), encoding="utf-8")
    return str(path)


def listar_experimentos_salvos() -> list:
    _garantir_dir()
    out = []
    for arquivo in sorted(DIR_EXPERIMENTOS.glob("*.json"), reverse=True):
        try:
            dados = _json.loads(arquivo.read_text(encoding="utf-8"))
            res = dados["resultados"]
            out.append({
                "arquivo": str(arquivo),
                "nome": dados.get("experiment_name", arquivo.stem),
                "timestamp": dados.get("timestamp", ""),
                "config": dados.get("config", {}),
                "acuracia_media": res["acuracia_media"],
                "pd": res["performance_dropping_rate"],
                "forgetting": res["esquecimento_medio"],
                "n_sessoes": len(res["historico_acuracias"]),
                "_dados": dados,
            })
        except Exception:
            continue
    return out


def plotar_curvas_comparacion(curvas: list, ax=None):
    """curvas: list of (label: str, acuracias: list)"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    colormap = plt.cm.tab10
    for i, (label, vals) in enumerate(curvas):
        xs = list(range(len(vals)))
        c = colormap(i / max(len(curvas) - 1, 1))
        ax.plot(xs, vals, marker="o", linewidth=2, markersize=6, color=c, label=label)
        ax.fill_between(xs, vals, alpha=0.08, color=c)
    ax.set_xlabel("Sessão", fontsize=12)
    ax.set_ylabel("Acurácia", fontsize=12)
    ax.set_title("Comparación de Acurácias entre Experimentos", fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    return ax


def renderizar_aba_configuracao():
    st.header("Configuração do Experimento")
    with st.form("form_configuracao"):
        col1, col2 = st.columns(2)
        with col1:
            dataset_escolhido = st.selectbox(
                "Dataset",
                ["Sintético (teste rápido)", "CIFAR-100", "PlantVillage", "PlantDoc"],
                help="Sintético gera dados aleatórios para testar o pipeline. "
                     "CIFAR-100, PlantVillage e PlantDoc devem estar baixados localmente.",
            )
            nome_dataset = (
                "sintetico"
                if "Sintético" in dataset_escolhido
                else "cifar100"
                if "CIFAR" in dataset_escolhido
                else "plantvillage"
                if "PlantVillage" in dataset_escolhido
                else "plantdoc"
            )
            pasta_dados = st.text_input(
                "Diretório dos dados brutos",
                value=os.path.join(str(Path(__file__).parent), "data", "raw"),
            )
            semente = st.number_input(
                "Semente aleatória", min_value=0, max_value=2**31 - 1, value=42,
            )
        with col2:
            numero_classes_base = st.number_input(
                "Classes na sessão base",
                min_value=2, max_value=100, value=20,
                help="Quantas classes diferentes compõem a sessão base.",
            )
            classes_por_sessao = st.number_input(
                "N (classes novas por safra)",
                min_value=1, max_value=20, value=5,
                help="N-way: número de novas doenças introduzidas em cada sessão.",
            )
            k_shot = st.number_input(
                "K (exemplos por classe nova)",
                min_value=1, max_value=50, value=5,
                help="K-shot: poucos exemplos disponíveis para cada nova doença.",
            )
            estrategia = st.selectbox(
                "Estratégia de atualização",
                ["media_simples", "media_movel_exponencial", "recalibracao_por_contagem"],
                help="Como os protótipos existentes são ajustados ao incorporar novas classes.",
            )
            incluir_hog = st.checkbox("Incluir descritores HOG", value=False)
        st.markdown("---")
        st.markdown(
            "**Importante:** Los datasets CIFAR-100, PlantVillage y PlantDoc deben estar "
            "descargados manualmente y colocados en el directorio de datos antes de cargarlos."
        )
        botao_configurar = st.form_submit_button("Carregar e Configurar", type="primary")
    if botao_configurar:
        with st.spinner("Carregando dataset..."):
            try:
                dados_brutos = carregar_dataset_bruto(nome_dataset, pasta_dados)
                st.success(
                    f"Dataset carregado: {dados_brutos['imagens'].shape[0]} imagens, "
                    f"{len(np.unique(dados_brutos['rotulos']))} classes."
                )
            except Exception as erro:
                st.error(f"Erro ao carregar dataset: {erro}")
                return
        with st.spinner("Dividindo sessões..."):
            sessoes = dividir_em_sessoes(
                dados_brutos["rotulos"],
                numero_classes_base,
                classes_por_sessao,
                semente,
            )
        nomes_dataset = obter_nomes_dataset(dataset_escolhido)
        nomes_classes_externos = list(nomes_dataset) if nomes_dataset else None
        gerenciador = GerenciadorDeSessoes(
            dados_brutos=dados_brutos,
            sessoes=sessoes,
            tamanho_alvo=(64, 64),
            incluir_hog=incluir_hog,
            semente_aleatoria=semente,
            nomes_classes_externos=nomes_classes_externos,
        )
        with st.spinner("Executando sessão base..."):
            resultado_base = gerenciador.executar_sessao_base()
        st.session_state.gerenciador = gerenciador
        st.session_state.sessao_atual = 0
        st.session_state.total_sessoes = gerenciador.obter_numero_sessoes_disponiveis()
        st.session_state.configurado = True
        st.session_state.resultados = gerenciador.obter_relatorio()
        st.session_state.indice_ultima_sessao = 0
        st.session_state.config_experimento = {
            "dataset": dataset_escolhido,
            "classes_base": numero_classes_base,
            "classes_por_sessao": classes_por_sessao,
            "k_shot": k_shot,
            "estrategia": estrategia,
            "incluir_hog": incluir_hog,
            "semente": semente,
        }
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Sessão base concluída", f"{len(sessoes['sessao_base'])} classes")
        col_b.metric("Acurácia inicial", f"{resultado_base['metricas']:.3f}")
        col_c.metric(
            "Sessões incrementais disponíveis",
            f"{len(sessoes['sessoes_incrementais'])}",
        )
        st.info(
            "Configuração concluída! Vá para a aba 'Execução Incremental' "
            "para simular a chegada de novas safras."
        )


def renderizar_aba_execucao():
    st.header("Nova Safra — Execução Incremental")
    if not st.session_state.configurado:
        st.warning("Configure o experimento na aba 'Configuração' primeiro.")
        return
    gerenciador = st.session_state.gerenciador
    sessao_atual = st.session_state.indice_ultima_sessao
    sessoes_incrementais = gerenciador.sessoes["sessoes_incrementais"]
    total_sessoes_inc = len(sessoes_incrementais)
    if sessao_atual >= total_sessoes_inc:
        st.success("Todas as sessões foram executadas! Veja os resultados na aba 'Resultados'.")
        return
    st.markdown(f"### Sessão incremental {sessao_atual + 1} de {total_sessoes_inc}")
    classes_nesta_sessao = sessoes_incrementais[sessao_atual]
    nomes = gerenciador.memoria.obter_mapeamento_classes_para_nomes()
    st.markdown("**Nuevas enfermedades/plagas de esta campaña:**")
    dados_plagas = []
    for classe in classes_nesta_sessao:
        clave = nomes.get(int(classe), f"Classe {classe}")
        info = GUIA_COMPLETA.get(clave, {})
        dados_plagas.append({
            "Nombre común": formatear_nombre_plaga(clave),
            "Cultivo": info.get("cultivo", "-"),
            "Tipo": NOMBRE_TIPO.get(info.get("tipo", ""), "Desconocido"),
            "Agente causal": info.get("agente_causal", "-"),
            "Síntomas": info.get("sintomas", "-"),
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(dados_plagas), width="stretch", hide_index=True)
    with st.expander("Ver imágenes de ejemplo de las nuevas clases"):
        for classe in classes_nesta_sessao:
            clave = nomes.get(int(classe), f"Classe {classe}")
            info = GUIA_COMPLETA.get(clave, {})
            dados_amostra = _preparar_dados_sessao(
                gerenciador.dados_brutos,
                [int(classe)],
                quantidade_por_classe=3,
                semente_aleatoria=gerenciador.semente_aleatoria + sessao_atual,
                modo_treino=False,
            )
            st.markdown(f"**{formatear_nombre_plaga(clave)}**")
            cols_imgs = st.columns(min(3, len(dados_amostra["imagens"])))
            for idx_col, (col_img, img) in enumerate(zip(cols_imgs, dados_amostra["imagens"][:3])):
                with col_img:
                    st.image(img, width=120, caption=f"Ejemplo {idx_col + 1}")
    k_shot = st.number_input(
        "K (exemplos por classe para esta sessão)",
        min_value=1,
        max_value=50,
        value=5,
        key=f"k_shot_{sessao_atual}",
    )
    estrategia = st.session_state.get("estrategia_atual", "media_simples")
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        botao_executar = st.button(
            f"Simular chegada da safra {sessao_atual + 1}",
            type="primary",
            disabled=st.session_state.executando,
        )
    if botao_executar:
        st.session_state.executando = True
        with st.spinner(f"Processando sessão {sessao_atual + 1}..."):
            try:
                resultado = gerenciador.executar_sessao_incremental(
                    sessao_atual,
                    quantidade_por_classe=k_shot,
                    estrategia_atualizacao=estrategia,
                )
                st.session_state.indice_ultima_sessao = sessao_atual + 1
                st.session_state.resultados = gerenciador.obter_relatorio()
                st.success(
                    f"Sessão {sessao_atual + 1} concluída! "
                    f"Acurácia acumulada: {resultado['metrica']:.3f}"
                )
                st.balloons()
            except Exception as erro:
                st.error(f"Erro durante execução: {erro}")
            finally:
                st.session_state.executando = False
    historico = st.session_state.resultados["historico_acuracias"]
    if len(historico) > 0:
        st.subheader("Curva de Acurácia (atualizada)")
        fig_ac, ax_ac = plt.subplots(figsize=(8, 4))
        plotar_curva_acuracia(historico, ax=ax_ac)
        st.pyplot(fig_ac)


def renderizar_aba_resultados():
    st.header("Resultados e Métricas")
    if st.session_state.resultados is None:
        st.warning("Nenhum resultado disponível. Execute ao menos a sessão base.")
        return
    resultados = st.session_state.resultados
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia Média", f"{resultados['acuracia_media']:.4f}")
    col2.metric(
        "Performance Dropping Rate",
        f"{resultados['performance_dropping_rate']:.4f}",
        delta_color="inverse",
        help="Diferença entre a acurácia da primeira e da última sessão. "
        "Valores altos indicam esquecimento catastrófico.",
    )
    col3.metric(
        "Esquecimento Médio",
        f"{resultados['esquecimento_medio']:.4f}",
        delta_color="inverse",
        help="Média da queda de acurácia por classe entre o pico histórico e o valor final.",
    )
    st.subheader("Curva de Acurácia por Sessão")
    fig_curva, ax_curva = plt.subplots(figsize=(10, 5))
    plotar_curva_acuracia(resultados["historico_acuracias"], ax=ax_curva)
    st.pyplot(fig_curva)
    gerenciador = st.session_state.gerenciador
    memoria = gerenciador.memoria
    nomes = memoria.obter_mapeamento_classes_para_nomes()
    rotulos_ordenados = memoria.obter_rotulos_ordenados()

    st.subheader("Diagnósticos disponibles en la memoria")
    dados_tabela_clases = []
    for r in rotulos_ordenados:
        clave = nomes.get(int(r), "")
        info = GUIA_COMPLETA.get(clave, {})
        tipo = info.get("tipo", "desconocido") if info else "desconocido"
        dados_tabela_clases.append({
            "ID": r,
            "Nombre común": formatear_nombre_plaga(clave),
            "Cultivo": info.get("cultivo", obtener_cultivo(clave)) if clave else "-",
            "Tipo": tipo.capitalize(),
        })
    if dados_tabela_clases:
        import pandas as pd
        st.dataframe(pd.DataFrame(dados_tabela_clases), width="stretch", hide_index=True)

    if len(resultados["historico_matrizes"]) > 0:
        st.subheader("Matriz de Confusión - Última Sesión")
        matriz = resultados["historico_matrizes"][-1]
        if matriz.size > 0:
            classes_nomes = [
                formatear_nombre_plaga(nomes.get(int(r), f"Classe {r}"))
                for r in rotulos_ordenados
            ]
            if matriz.shape[0] == len(classes_nomes):
                fig_matriz, ax_matriz = plt.subplots(figsize=(10, 9))
                plotar_matriz_confusao(matriz, classes_nomes, ax=ax_matriz)
                st.pyplot(fig_matriz)
            else:
                st.info("Matriz de confusión disponible (dimensiones incompatibles).")
    # ---- Guardar experimento ----
    with st.expander("Guardar experimento"):
        nome_salvar = st.text_input(
            "Nombre del experimento",
            value=st.session_state.config_experimento.get("dataset", "experimento"),
            key="nome_salvar_input",
        )
        if st.button("Guardar experimento", type="primary"):
            path = salvar_experimento(
                nome_salvar,
                st.session_state.config_experimento,
                resultados,
            )
            st.success(f"Experimento guardado: `{path}`")

    # ---- Comparar experimentos ----
    with st.expander("Comparar con experimentos guardados"):
        salvos = listar_experimentos_salvos()
        if not salvos:
            st.info("No hay experimentos guardados todavía.")
        else:
            opcoes = {
                f"{e['nome']}  ({e['timestamp']})  |  μ={e['acuracia_media']:.3f}  PD={e['pd']:.3f}  F={e['forgetting']:.3f}": e
                for e in salvos
            }
            selecionados = st.multiselect(
                "Seleccionar experimentos para comparar",
                options=list(opcoes.keys()),
                default=[],
            )
            if selecionados and st.button("Mostrar comparación"):
                curvas = [
                    (
                        f"Actual ({st.session_state.config_experimento.get('dataset', '?')})",
                        resultados["historico_acuracias"],
                    )
                ]
                dados_tabela = [
                    {
                        "Experimento": "Actual",
                        "Dataset": st.session_state.config_experimento.get("dataset", "?"),
                        "Acurácia media": f"{resultados['acuracia_media']:.4f}",
                        "PD Rate": f"{resultados['performance_dropping_rate']:.4f}",
                        "Esq. medio": f"{resultados['esquecimento_medio']:.4f}",
                        "Sesiones": len(resultados["historico_acuracias"]),
                    }
                ]
                for chave in selecionados:
                    e = opcoes[chave]
                    curvas.append((e["nome"], e["_dados"]["resultados"]["historico_acuracias"]))
                    dados_tabela.append({
                        "Experimento": e["nome"],
                        "Dataset": e["config"].get("dataset", "?"),
                        "Acurácia media": f"{e['acuracia_media']:.4f}",
                        "PD Rate": f"{e['pd']:.4f}",
                        "Esq. medio": f"{e['forgetting']:.4f}",
                        "Sesiones": e["n_sessoes"],
                    })
                st.subheader("Curvas comparadas")
                fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
                plotar_curvas_comparacion(curvas, ax=ax_comp)
                st.pyplot(fig_comp)
                st.subheader("Tabla comparativa")
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(dados_tabela),
                    width="stretch",
                    hide_index=True,
                )

    with st.expander("Exportar resultados"):
        import pandas as pd
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            datos_csv = {"Sesión": [], "Precisión": []}
            for indice, valor in enumerate(resultados["historico_acuracias"]):
                datos_csv["Sesión"].append(indice)
                datos_csv["Precisión"].append(valor)
            df_csv = pd.DataFrame(datos_csv)
            csv_bytes = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="CSV (precisión por sesión)",
                data=csv_bytes,
                file_name="fscil_resultados_precision.csv",
                mime="text/csv",
            )
        with col_exp2:
            resultados_json = dict(resultados)
            resultados_json["historico_matrizes"] = [
                m.tolist() for m in resultados_json["historico_matrizes"]
            ]
            if hasattr(resultados_json.get("historico_por_classe"), "tolist"):
                resultados_json["historico_por_classe"] = resultados_json["historico_por_classe"].tolist()
            json_bytes = dicionario_para_json(resultados_json).encode("utf-8")
            st.download_button(
                label="JSON (completo)",
                data=json_bytes,
                file_name="fscil_resultados_completos.json",
                mime="application/json",
            )


def renderizar_aba_espaco():
    st.header("Espaço de Características")
    if not st.session_state.configurado:
        st.warning("Configure o experimento primeiro.")
        return
    gerenciador = st.session_state.gerenciador
    memoria = gerenciador.memoria
    prototipos = memoria.obter_prototipos()
    if prototipos is None:
        st.warning("Nenhum protótipo disponível.")
        return
    classes_vistas = gerenciador.total_classes_vistas
    dados_teste = _preparar_dados_sessao(
        gerenciador.dados_brutos,
        classes_vistas,
        quantidade_por_classe=20,
        semente_aleatoria=gerenciador.semente_aleatoria,
        modo_treino=False,
    )
    with st.spinner("Extraindo características para visualização..."):
        preprocessado = pipeline_completo_preprocessamento(
            dados_teste["imagens"], tamanho_alvo=gerenciador.tamanho_alvo
        )
        caracteristicas, _ = extrair_todas_caracteristicas(
            preprocessado["imagens_cinza"],
            preprocessado["mascaras_binarias"],
            incluir_hog=gerenciador.incluir_hog,
            escalonador=gerenciador.escalonador,
        )
    rotulos_prototipos = memoria.obter_rotulos_ordenados()
    nomes_classes = memoria.obter_mapeamento_classes_para_nomes()
    fig_espaco, ax_espaco = plt.subplots(figsize=(11, 8))
    plotar_espaco_caracteristicas(
        caracteristicas,
        dados_teste["rotulos"],
        prototipos,
        rotulos_prototipos,
        nomes_classes,
        ax=ax_espaco,
    )
    st.pyplot(fig_espaco)
    st.caption(
        "Cada ponto é uma imagem projetada em 2 componentes principais (PCA). "
        "Os 'X' vermelhos são os protótipos de cada classe."
    )


def renderizar_aba_demo():
    st.header("Diagnóstico de Campo")
    st.markdown(
        "Faça upload de uma foto de folha para diagnóstico. "
        "O sistema extrairá as características e comparará com os protótipos conhecidos."
    )
    if not st.session_state.configurado:
        st.warning("Configure e execute o experimento primeiro.")
        return
    gerenciador = st.session_state.gerenciador
    memoria = gerenciador.memoria
    nomes_classes = memoria.obter_mapeamento_classes_para_nomes()
    arquivo_upload = st.file_uploader(
        "Escolha uma imagem de folha",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )
    if arquivo_upload is not None:
        from skimage.io import imread
        from PIL import Image

        imagem_pil = Image.open(arquivo_upload).convert("RGB")
        imagem_np = np.array(imagem_pil)
        st.image(imagem_np, caption="Imagem enviada", width=300)
        with st.spinner("Extraindo características..."):
            preprocessado = pipeline_completo_preprocessamento(
                np.expand_dims(imagem_np, axis=0),
                tamanho_alvo=gerenciador.tamanho_alvo,
            )
            caracteristicas, _ = extrair_todas_caracteristicas(
                preprocessado["imagens_cinza"],
                preprocessado["mascaras_binarias"],
                incluir_hog=gerenciador.incluir_hog,
                escalonador=gerenciador.escalonador,
            )
        if memoria.prototipos is not None:
            distancias, indices_ordenados = memoria.obter_distancias(caracteristicas)
            distancias_ordenadas = distancias[0, indices_ordenados[0]]
            st.subheader("Resultado del Diagnóstico")
            classe_predita = memoria.classificar(caracteristicas)[0]
            clave_predita = nomes_classes.get(
                int(classe_predita), f"Classe {classe_predita}"
            )
            info_predita = GUIA_COMPLETA.get(clave_predita, {})
            distancia_minima = distancias_ordenadas[0]
            confianca = 1.0 / (1.0 + distancia_minima)
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Diagnóstico", formatear_nombre_plaga(clave_predita))
            col_d2.metric("Confianza (1/(1+d))", f"{confianca:.4f}")
            col_d3.metric("Tipo", NOMBRE_TIPO.get(info_predita.get("tipo", ""), "-"))
            if info_predita:
                with st.expander("Ver detalles de esta plaga/enfermedad", expanded=True):
                    st.markdown(f"**Agente causal:** {info_predita.get('agente_causal', '-')}")
                    st.markdown(f"**Síntomas:** {info_predita.get('sintomas', '-')}")
                    st.markdown(f"**Tratamiento sugerido:** {info_predita.get('tratamiento', '-')}")
            st.markdown("#### Top-3 clases más cercanas")
            dados_tabela = []
            for rank in range(min(3, len(distancias_ordenadas))):
                indice = indices_ordenados[0, rank]
                rotulo = gerenciador.memoria.mapeamento_indice_para_classe[indice]
                clave = nomes_classes.get(int(rotulo), f"Classe {rotulo}")
                dados_tabela.append(
                    {
                        "Rank": rank + 1,
                        "Nombre común": formatear_nombre_plaga(clave),
                        "Distancia Euclidiana": f"{distancias_ordenadas[rank]:.4f}",
                    }
                )
            import pandas as pd

            st.table(pd.DataFrame(dados_tabela))
            if confianca < 0.3:
                st.warning(
                    "Confianza baja en todas las clases conocidas. "
                    "Podría ser una enfermedad/plaga nueva. Considere registrar una nueva clase."
                )
                if st.button("Registrar como nueva enfermedad", type="secondary"):
                    st.info(
                        "Esta funcionalidad permite al usuario proporcionar K fotos "
                        "y confirmar el diagnóstico para incorporar la nueva clase al sistema."
                    )
        else:
            st.warning("Memória de protótipos vazia. Execute a sessão base primeiro.")


def renderizar_aba_guia():
    st.header("Guía de Plagas y Enfermedades Foliares")
    st.markdown(
        "Base de datos de referencia con información sobre las enfermedades, plagas "
        "y estados sanos disponibles en el dataset **PlantVillage**. "
        "Utilice esta guía para identificar visualmente los diagnósticos que su sistema puede reconocer."
    )
    estadisticas = obtener_estadisticas()
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    col_g1.metric("Total clases", estadisticas["total_clases"])
    col_g2.metric("Cultivos", estadisticas["total_cultivos"])
    col_g3.metric("Enfermedades", estadisticas["enfermedades"])
    col_g4.metric("Plagas", estadisticas["plagas"])
    tab_por_cultivo, tab_por_tipo, tab_tabela = st.tabs(
        ["Por cultivo", "Por tipo", "Tabla completa"]
    )
    with tab_por_cultivo:
        for cultivo, clases in listar_por_cultivo().items():
            with st.expander(f"{cultivo} ({len(clases)} clases)"):
                for item in clases:
                    tipo_label = NOMBRE_TIPO.get(item.get("tipo", ""), "Desconocido")
                    st.markdown(
                        f"- **{item['nombre_comun']}** "
                        f"({tipo_label}) — {item.get('sintomas', '')[:100]}..."
                    )
    with tab_por_tipo:
        for tipo, clases in listar_por_tipo().items():
            if not clases:
                continue
            tipo_label = NOMBRE_TIPO.get(tipo, tipo)
            with st.expander(f"{tipo_label} ({len(clases)} clases)"):
                for item in clases:
                    st.markdown(
                        f"- **{item['nombre_comun']}** (cultivo: {item['cultivo']}) — "
                        f"{item.get('agente_causal', '')}"
                    )
    with tab_tabela:
        import pandas as pd
        filas = []
        for clave, info in GUIA_COMPLETA.items():
            filas.append({
                "Clave dataset": clave,
                "Nombre común": info["nombre_comun"],
                "Cultivo": info["cultivo"],
                "Tipo": NOMBRE_TIPO.get(info.get("tipo", ""), info.get("tipo", "")),
                "Agente causal": info["agente_causal"],
                "Síntomas": info["sintomas"][:80] + "...",
                "Tratamiento": info["tratamiento"][:80] + "...",
            })
        st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)


def main():
    inicializar_estado()
    tab_config, tab_exec, tab_result, tab_espaco, tab_demo, tab_guia = st.tabs(
        [
            "Configuración",
            "Ejecución Incremental",
            "Resultados",
            "Espacio de Características",
            "Diagnóstico de Campo",
            "Guía de Plagas",
        ]
    )
    with tab_config:
        renderizar_aba_configuracao()
    with tab_exec:
        renderizar_aba_execucao()
    with tab_result:
        renderizar_aba_resultados()
    with tab_espaco:
        renderizar_aba_espaco()
    with tab_demo:
        renderizar_aba_demo()
    with tab_guia:
        renderizar_aba_guia()


if __name__ == "__main__":
    main()
