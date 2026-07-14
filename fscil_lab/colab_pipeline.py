#!/usr/bin/env python3
"""
Script autónomo para ejecutar el pipeline FSCIL en Google Colab.

Uso en Colab:
1. Sube este archivo junto con la carpeta `src/` y tus datasets a Google Drive.
2. En una celda de Colab:
    !pip install numpy scipy scikit-image scikit-learn opencv-python pandas matplotlib
    import sys; sys.path.insert(0, '/content/drive/MyDrive/fscil_lab')
    from colab_pipeline import ejecutar_experimento
    ejecutar_experimento(
        dataset="plantvillage",
        data_dir="/content/drive/MyDrive/fscil_lab/data/raw",
        output_dir="/content/drive/MyDrive/fscil_lab/resultados",
        num_classes_base=20, n_way=5, k_shot=5,
        estrategia="media_simples", incluir_hog=False, semilla=42
    )

3. Descarga el archivo JSON generado en `output_dir`.
4. En la app local, ve a Resultados > "Importar resultados de Colab" y súbelo.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

import numpy as np


RUTA_SCRIPTS = Path(__file__).parent.resolve()
if str(RUTA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RUTA_SCRIPTS))

from src.dataset_loader import carregar_dataset_bruto, dividir_em_sessoes, preparar_dados_sessao
from src.preprocessing import pipeline_completo_preprocessamento
from src.feature_extraction import extrair_todas_caracteristicas
from src.prototype_memory import MemoriaDePrototipos
from src.session_manager import GerenciadorDeSessoes
from src.metrics import gerar_relatorio_sessoes
from src.utils import dicionario_para_json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def serializar_memoria(memoria: MemoriaDePrototipos) -> dict:
    return {
        "prototipos": memoria.prototipos.tolist() if memoria.prototipos is not None else None,
        "mapeamento_classe_para_indice": {str(k): v for k, v in memoria.mapeamento_classe_para_indice.items()},
        "mapeamento_indice_para_classe": {str(k): v for k, v in memoria.mapeamento_indice_para_classe.items()},
        "nomes_classes": {str(k): v for k, v in memoria.nomes_classes.items()},
        "historico_contagem_amostras": {str(k): v for k, v in memoria.historico_contagem_amostras.items()},
    }


def serializar_escalonador(escalonador) -> dict:
    if escalonador is None:
        return None
    out = {"n_features_in_": escalonador.n_features_in_}
    for attr in ("mean_", "scale_", "var_", "n_samples_seen_"):
        val = getattr(escalonador, attr, None)
        if val is not None:
            out[attr] = val.tolist() if hasattr(val, "tolist") else val
    return out


def reconstruir_memoria(dados: dict) -> MemoriaDePrototipos:
    memoria = MemoriaDePrototipos()
    if dados.get("prototipos") is not None:
        memoria.prototipos = np.array(dados["prototipos"])
    memoria.mapeamento_classe_para_indice = {int(k): v for k, v in dados.get("mapeamento_classe_para_indice", {}).items()}
    memoria.mapeamento_indice_para_classe = {int(k): v for k, v in dados.get("mapeamento_indice_para_classe", {}).items()}
    memoria.nomes_classes = {int(k): v for k, v in dados.get("nomes_classes", {}).items()}
    memoria.historico_contagem_amostras = {int(k): v for k, v in dados.get("historico_contagem_amostras", {}).items()}
    return memoria


def reconstruir_escalonador(dados: dict):
    if dados is None:
        return None
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    for attr in ("mean_", "scale_", "var_", "n_features_in_", "n_samples_seen_"):
        val = dados.get(attr)
        if val is not None:
            setattr(scaler, attr, np.array(val) if isinstance(val, list) else val)
    return scaler


def mapear_nombres_externos(dataset: str, nomes_dataset: list) -> dict:
    mapping = {}
    if nomes_dataset:
        for i, nome in enumerate(nomes_dataset):
            mapping[i] = nome
    return mapping


def ejecutar_experimento(
    dataset: str = "sintetico",
    data_dir: str = "data/raw",
    output_dir: str = "resultados",
    num_classes_base: int = 20,
    n_way: int = 5,
    k_shot: int = 5,
    estrategia: str = "media_simples",
    incluir_hog: bool = False,
    semilla: int = 42,
    nombre_experimento: str = None,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_map = {
        "sintetico": "Sintético (teste rápido)",
        "cifar100": "CIFAR-100",
        "plantvillage": "PlantVillage",
        "plantdoc": "PlantDoc",
    }

    print(f"[ColabPipeline] Cargando dataset '{dataset}' desde {data_dir}...")
    dados_brutos = carregar_dataset_bruto(dataset, data_dir)
    print(f"  -> {dados_brutos['imagens'].shape[0]} imágenes, {len(np.unique(dados_brutos['rotulos']))} clases")

    print(f"[ColabPipeline] Dividendo en sesiones (base={num_classes_base}, N-way={n_way})...")
    sessoes = dividir_em_sessoes(
        dados_brutos["rotulos"],
        num_classes_base,
        n_way,
        semilla,
    )

    nomes_dataset_list = dados_brutos.get("nomes_classes")
    gerenciador = GerenciadorDeSessoes(
        dados_brutos=dados_brutos,
        sessoes=sessoes,
        tamanho_alvo=(64, 64),
        incluir_hog=incluir_hog,
        semente_aleatoria=semilla,
        nomes_classes_externos=nomes_dataset_list,
    )

    print("[ColabPipeline] Ejecutando sesión base...")
    gerenciador.executar_sessao_base()
    print(f"  -> Acurácia inicial: {gerenciador.historico_acuracias[-1]:.4f}")

    total_inc = len(sessoes["sessoes_incrementais"])
    print(f"[ColabPipeline] Ejecutando {total_inc} sesiones incrementales...")
    for idx in range(total_inc):
        gerenciador.executar_sessao_incremental(
            idx,
            quantidade_por_classe=k_shot,
            estrategia_atualizacao=estrategia,
        )
        print(f"  -> Sesión {idx+1}/{total_inc}: acurácia = {gerenciador.historico_acuracias[-1]:.4f}")

    resultados = gerar_relatorio_sessoes(
        gerenciador.historico_acuracias,
        gerenciador.historico_matrizes,
        gerenciador.historico_por_classe,
    )

    nombre = nombre_experimento or f"{dataset}_b{num_classes_base}_n{n_way}_k{k_shot}_{estrategia}"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in nombre).strip()[:60]
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}__{safe_name}.json"

    memoria = gerenciador.memoria
    package = {
        "experiment_name": nombre,
        "timestamp": timestamp,
        "origen": "colab",
        "config": {
            "dataset": dataset_map.get(dataset, dataset),
            "classes_base": num_classes_base,
            "classes_por_sessao": n_way,
            "k_shot": k_shot,
            "estrategia": estrategia,
            "incluir_hog": incluir_hog,
            "semente": semilla,
        },
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
        "memoria": serializar_memoria(memoria),
        "escalonador": serializar_escalonador(gerenciador.escalonador),
        "total_classes_vistas": gerenciador.total_classes_vistas,
    }

    ruta_archivo = output_path / filename
    ruta_archivo.write_text(
        json.dumps(package, cls=NumpyEncoder, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[ColabPipeline] Experimento guardado en: {ruta_archivo}")
    print(f"[ColabPipeline] Acurácia media: {resultados['acuracia_media']:.4f}")
    print(f"[ColabPipeline] PD Rate: {resultados['performance_dropping_rate']:.4f}")
    print(f"[ColabPipeline] Forgetting: {resultados['esquecimento_medio']:.4f}")
    print("\n[ColabPipeline] ¡Listo! Descarga el archivo JSON y cárgalo en la app local.")

    return str(ruta_archivo)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecutar pipeline FSCIL para Colab")
    parser.add_argument("--dataset", default="sintetico", choices=["sintetico", "cifar100", "plantvillage", "plantdoc"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="resultados")
    parser.add_argument("--classes-base", type=int, default=20)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--estrategia", default="media_simples", choices=["media_simples", "media_movel_exponencial", "recalibracao_por_contagem"])
    parser.add_argument("--hog", action="store_true")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--nombre", default=None)
    args = parser.parse_args()

    ejecutar_experimento(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_classes_base=args.classes_base,
        n_way=args.n_way,
        k_shot=args.k_shot,
        estrategia=args.estrategia,
        incluir_hog=args.hog,
        semilla=args.semilla,
        nombre_experimento=args.nombre,
    )
