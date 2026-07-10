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
from src.dataset_loader import preparar_dados_sessao as _preparar_dados_sessao
from src.utils import dicionario_para_json

st.set_page_config(
    page_title="FSCIL-Lab — Diagnóstico Fitossanitário Incremental",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("FSCIL-Lab")
st.sidebar.markdown("Diagnóstico incremental de doenças foliares")
st.sidebar.markdown("---")


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
    }
    for chave, valor in chaves_padrao.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def obter_nomes_dataset(dataset: str) -> list:
    exemplos = {
        "CIFAR-100 (protótipo)": [
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


def renderizar_aba_configuracao():
    st.header("Configuração do Experimento")
    with st.form("form_configuracao"):
        col1, col2 = st.columns(2)
        with col1:
            dataset_escolhido = st.selectbox(
                "Dataset",
                ["CIFAR-100 (protótipo)", "PlantVillage", "PlantDoc"],
                help="CIFAR-100 é usado como dataset protótipo para testes.",
            )
            nome_dataset = (
                "cifar100"
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
            "**Nota:** CIFAR-100 será baixado automaticamente via TensorFlow/Keras. "
            "PlantVillage e PlantDoc requerem download manual."
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
                st.info(
                    "Dica: para CIFAR-100, instale tensorflow: pip install tensorflow. "
                    "Para PlantVillage/PlantDoc, baixe manualmente e ajuste o diretório."
                )
                return
        with st.spinner("Dividindo sessões..."):
            sessoes = dividir_em_sessoes(
                dados_brutos["rotulos"],
                numero_classes_base,
                classes_por_sessao,
                semente,
            )
        nomes_dataset = obter_nomes_dataset(dataset_escolhido)
        nomes_classes_externos = {}
        if nomes_dataset:
            for indice, nome in enumerate(nomes_dataset):
                nomes_classes_externos[indice] = nome
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
    st.markdown("**Novas doenças desta safra:**")
    for classe in classes_nesta_sessao:
        nome = nomes.get(int(classe), f"Classe {classe}")
        st.markdown(f"- {nome} (ID {classe})")
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
    if len(resultados["historico_matrizes"]) > 0:
        st.subheader("Matriz de Confusão da Última Sessão")
        matriz = resultados["historico_matrizes"][-1]
        if matriz.size > 0:
            gerenciador = st.session_state.gerenciador
            rotulos_ordenados = gerenciador.memoria.obter_rotulos_ordenados()
            nomes = gerenciador.memoria.obter_mapeamento_classes_para_nomes()
            classes_nomes = [
                nomes.get(int(r), f"Classe {r}") for r in rotulos_ordenados
            ]
            if matriz.shape[0] == len(classes_nomes):
                fig_matriz, ax_matriz = plt.subplots(figsize=(9, 8))
                plotar_matriz_confusao(matriz, classes_nomes, ax=ax_matriz)
                st.pyplot(fig_matriz)
            else:
                st.info("Matriz de confusão disponível (dimensões incompatíveis para rótulos).")
    with st.expander("Dados brutos (JSON)"):
        st.code(dicionario_para_json(resultados), language="json")


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
            st.subheader("Resultado do Diagnóstico")
            classe_predita = memoria.classificar(caracteristicas)[0]
            nome_predito = nomes_classes.get(
                int(classe_predita), f"Classe {classe_predita}"
            )
            distancia_minima = distancias_ordenadas[0]
            confianca = 1.0 / (1.0 + distancia_minima)
            st.metric("Diagnóstico", nome_predito)
            st.metric("Confiança (1/(1+d))", f"{confianca:.4f}")
            st.markdown("#### Top-3 classes mais próximas")
            dados_tabela = []
            for rank in range(min(3, len(distancias_ordenadas))):
                indice = indices_ordenados[0, rank]
                rotulo = gerenciador.memoria.mapeamento_indice_para_classe[indice]
                nome = nomes_classes.get(int(rotulo), f"Classe {rotulo}")
                dados_tabela.append(
                    {
                        "Rank": rank + 1,
                        "Classe": nome,
                        "Distância Euclidiana": f"{distancias_ordenadas[rank]:.4f}",
                    }
                )
            import pandas as pd

            st.table(pd.DataFrame(dados_tabela))
            if confianca < 0.3:
                st.warning(
                    "Confiança baixa em todas as classes conhecidas. "
                    "Pode ser uma doença nova! Considere cadastrá-la."
                )
                if st.button("Cadastrar como doença nova", type="secondary"):
                    st.info(
                        "Funcionalidade de cadastro de nova classe seria acionada aqui. "
                        "No fluxo real, o usuário forneceria K fotos e confirmaria o diagnóstico."
                    )
        else:
            st.warning("Memória de protótipos vazia. Execute a sessão base primeiro.")


def main():
    inicializar_estado()
    tab_config, tab_exec, tab_result, tab_espaco, tab_demo = st.tabs(
        [
            "Configuração",
            "Execução Incremental",
            "Resultados",
            "Espaço de Características",
            "Diagnóstico de Campo",
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


if __name__ == "__main__":
    main()
