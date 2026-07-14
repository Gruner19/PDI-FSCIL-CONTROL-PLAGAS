#!/usr/bin/env python3
"""
Servidor Flask para ejecutar el pipeline FSCIL en Google Colab con ngrok.

Uso en Colab (Google Colab notebook):

    # Celda 1: Montar Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # Celda 2: Instalar
    !pip install numpy scipy scikit-image scikit-learn opencv-python pandas matplotlib flask pyngrok pillow

    # Celda 3: Ejecutar servidor
    import sys; sys.path.insert(0, '/content/drive/MyDrive/fscil_lab')
    from colab_api import iniciar_servidor
    iniciar_servidor(data_dir='/content/drive/MyDrive/fscil_lab/data/raw')

Luego en la app local:
    Sidebar > Conexión Colab > pegar la URL ngrok que aparece
    Configuración > llenar formulario > "Ejecutar en Colab"
    Diagnóstico de Campo > activar "Usar Colab" > subir imagen
"""

import sys
import json
import io
import base64
import os
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

RUTA_SCRIPTS = Path(__file__).parent.resolve()
if str(RUTA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RUTA_SCRIPTS))

from src.dataset_loader import carregar_dataset_bruto, dividir_em_sessoes
from src.preprocessing import pipeline_completo_preprocessamento
from src.feature_extraction import extrair_todas_caracteristicas
from src.session_manager import GerenciadorDeSessoes
from src.metrics import gerar_relatorio_sessoes

app = Flask(__name__)
DATA_DIR = "data/raw"
gerenciador = None
resultados = None
config_actual = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def _serializar_resultados():
    if resultados is None:
        return None
    return {
        "historico_acuracias": list(resultados["historico_acuracias"]),
        "acuracia_media": float(resultados["acuracia_media"]),
        "performance_dropping_rate": float(resultados["performance_dropping_rate"]),
        "esquecimento_medio": float(resultados["esquecimento_medio"]),
        "historico_matrizes": [
            m.tolist() if hasattr(m, "tolist") else m
            for m in resultados["historico_matrizes"]
        ],
    }


def _serializar_memoria():
    if gerenciador is None:
        return None
    mem = gerenciador.memoria
    return {
        "prototipos": mem.prototipos.tolist() if mem.prototipos is not None else None,
        "mapeamento_classe_para_indice": {str(k): v for k, v in mem.mapeamento_classe_para_indice.items()},
        "mapeamento_indice_para_classe": {str(k): v for k, v in mem.mapeamento_indice_para_classe.items()},
        "nomes_classes": {str(k): v for k, v in mem.nomes_classes.items()},
        "historico_contagem_amostras": {str(k): v for k, v in mem.historico_contagem_amostras.items()},
    }


def _serializar_escalonador():
    if gerenciador is None or gerenciador.escalonador is None:
        return None
    esc = gerenciador.escalonador
    out = {"n_features_in_": esc.n_features_in_}
    for attr in ("mean_", "scale_", "var_", "n_samples_seen_"):
        val = getattr(esc, attr, None)
        if val is not None:
            out[attr] = val.tolist() if hasattr(val, "tolist") else val
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "experimento_cargado": gerenciador is not None,
    })


def _serializar_sessoes_info():
    """Devuelve info de todas las sesiones (base + incrementales) para el frontend."""
    if gerenciador is None:
        return None
    sessoes = gerenciador.sessoes
    info = []
    info.append({
        "tipo": "base",
        "indice": 0,
        "classes": [int(c) for c in sessoes["sessao_base"]],
        "ejecutada": gerenciador.numero_sessoes_executadas >= 1,
    })
    for i, cls in enumerate(sessoes["sessoes_incrementais"]):
        ejecutada = (i + 1) < gerenciador.numero_sessoes_executadas
        info.append({
            "tipo": "incremental",
            "indice": i + 1,
            "classes": [int(c) for c in cls],
            "ejecutada": ejecutada,
        })
    return info


def _estado_completo():
    """Ensambla el objeto de estado completo para el frontend."""
    if gerenciador is None:
        return {"status": "error", "message": "No hay experimento cargado"}
    return {
        "status": "ok",
        "config": config_actual,
        "sessione_info": _serializar_sessoes_info(),
        "resultados": _serializar_resultados(),
        "memoria": _serializar_memoria(),
        "escalonador": _serializar_escalonador(),
        "total_classes_vistas": [int(c) for c in gerenciador.total_classes_vistas],
        "sesion_actual": gerenciador.numero_sessoes_executadas - 1,
        "total_sesiones": 1 + len(gerenciador.sessoes["sessoes_incrementais"]),
    }


@app.route("/iniciar", methods=["POST"])
def iniciar():
    """Carga dataset, divide sesiones y ejecuta solo la sesión base."""
    global gerenciador, resultados, config_actual, DATA_DIR
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Cuerpo JSON requerido"}), 400

        dataset = data.get("dataset", "sintetico")
        data_dir = data.get("data_dir", DATA_DIR)
        num_classes_base = data.get("classes_base", 20)
        n_way = data.get("n_way", 5)
        incluir_hog = data.get("incluir_hog", False)
        semilla = data.get("semilla", 42)

        print(f"[ColabAPI] Iniciando: dataset='{dataset}', base={num_classes_base}, N-way={n_way}")
        dados_brutos = carregar_dataset_bruto(dataset, data_dir)
        sessoes = dividir_em_sessoes(dados_brutos["rotulos"], num_classes_base, n_way, semilla)
        nomes_dataset_list = dados_brutos.get("nomes_classes")

        gerenciador = GerenciadorDeSessoes(
            dados_brutos=dados_brutos,
            sessoes=sessoes,
            tamanho_alvo=(64, 64),
            incluir_hog=incluir_hog,
            semente_aleatoria=semilla,
            nomes_classes_externos=nomes_dataset_list,
        )

        gerenciador.executar_sessao_base()
        print(f"[ColabAPI] Sesión base ejecutada. Acurácia: {gerenciador.historico_acuracias[-1]:.4f}")

        dataset_map = {
            "sintetico": "Sintético (teste rápido)", "cifar100": "CIFAR-100",
            "plantvillage": "PlantVillage", "plantdoc": "PlantDoc",
        }
        config_actual = {
            "dataset": dataset_map.get(dataset, dataset),
            "classes_base": num_classes_base,
            "classes_por_sessao": n_way,
            "incluir_hog": incluir_hog,
            "semente": semilla,
        }

        resultados = gerar_relatorio_sessoes(
            gerenciador.historico_acuracias,
            gerenciador.historico_matrizes,
            gerenciador.historico_por_classe,
        )

        return jsonify(_estado_completo())

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/ejecutar-sesion", methods=["POST"])
def ejecutar_sesion():
    """Ejecuta la siguiente sesión incremental (una por llamada)."""
    global resultados
    if gerenciador is None:
        return jsonify({"status": "error", "message": "Ejecute /iniciar primero"}), 400

    try:
        data = request.get_json() or {}
        k_shot_local = data.get("k_shot", 5)
        estrategia_atual = data.get("estrategia", "media_simples")

        n_inc = len(gerenciador.sessoes["sessoes_incrementais"])
        ya_ejecutadas = gerenciador.numero_sessoes_executadas - 1  # 0 = base ejecutada
        if ya_ejecutadas >= n_inc:
            return jsonify({
                "status": "completado",
                "message": "Todas las sesiones incrementales ya fueron ejecutadas",
                "estado": _estado_completo(),
            })

        idx = ya_ejecutadas
        print(f"[ColabAPI] Ejecutando sesión incremental {idx+1}/{n_inc} (K={k_shot_local})...")
        gerenciador.executar_sessao_incremental(
            idx,
            quantidade_por_classe=k_shot_local,
            estrategia_atualizacao=estrategia_atual,
        )
        print(f"  -> Acurácia: {gerenciador.historico_acuracias[-1]:.4f}")

        resultados = gerar_relatorio_sessoes(
            gerenciador.historico_acuracias,
            gerenciador.historico_matrizes,
            gerenciador.historico_por_classe,
        )

        return jsonify(_estado_completo())

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/estado", methods=["GET"])
def estado():
    """Devuelve el estado actual del experimento."""
    if gerenciador is None:
        return jsonify({"status": "error", "message": "No hay experimento cargado"}), 400
    return jsonify(_estado_completo())


@app.route("/analizar", methods=["POST"])
def analizar():
    global gerenciador, resultados, config_actual, DATA_DIR
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Cuerpo JSON requerido"}), 400

        dataset = data.get("dataset", "sintetico")
        data_dir = data.get("data_dir", DATA_DIR)
        num_classes_base = data.get("classes_base", 20)
        n_way = data.get("n_way", 5)
        k_shot = data.get("k_shot", 5)
        estrategia = data.get("estrategia", "media_simples")
        incluir_hog = data.get("incluir_hog", False)
        semilla = data.get("semilla", 42)

        print(f"[ColabAPI] Cargando dataset '{dataset}' desde {data_dir}...")
        dados_brutos = carregar_dataset_bruto(dataset, data_dir)
        n_imgs = dados_brutos["imagens"].shape[0]
        n_cls = len(np.unique(dados_brutos["rotulos"]))
        print(f"[ColabAPI]  -> {n_imgs} imágenes, {n_cls} clases")

        sessoes = dividir_em_sessoes(dados_brutos["rotulos"], num_classes_base, n_way, semilla)
        nomes_dataset_list = dados_brutos.get("nomes_classes")

        gerenciador = GerenciadorDeSessoes(
            dados_brutos=dados_brutos,
            sessoes=sessoes,
            tamanho_alvo=(64, 64),
            incluir_hog=incluir_hog,
            semente_aleatoria=semilla,
            nomes_classes_externos=nomes_dataset_list,
        )

        print("[ColabAPI] Sesión base...")
        gerenciador.executar_sessao_base()
        ac_base = gerenciador.historico_acuracias[-1]
        print(f"[ColabAPI]  -> Acurácia: {ac_base:.4f}")

        total_inc = len(sessoes["sessoes_incrementais"])
        print(f"[ColabAPI] {total_inc} sesiones incrementales...")
        for idx in range(total_inc):
            gerenciador.executar_sessao_incremental(
                idx, quantidade_por_classe=k_shot, estrategia_atualizacao=estrategia,
            )
            print(f"  -> {idx+1}/{total_inc}: acurácia = {gerenciador.historico_acuracias[-1]:.4f}")

        resultados = gerar_relatorio_sessoes(
            gerenciador.historico_acuracias,
            gerenciador.historico_matrizes,
            gerenciador.historico_por_classe,
        )

        dataset_map = {
            "sintetico": "Sintético (teste rápido)", "cifar100": "CIFAR-100",
            "plantvillage": "PlantVillage", "plantdoc": "PlantDoc",
        }
        config_actual = {
            "dataset": dataset_map.get(dataset, dataset),
            "classes_base": num_classes_base,
            "classes_por_sessao": n_way,
            "k_shot": k_shot,
            "estrategia": estrategia,
            "incluir_hog": incluir_hog,
            "semente": semilla,
        }

        respuesta = {
            "status": "ok",
            "resultados": _serializar_resultados(),
            "config": config_actual,
            "memoria": _serializar_memoria(),
            "escalonador": _serializar_escalonador(),
            "total_classes_vistas": gerenciador.total_classes_vistas,
        }
        print(f"[ColabAPI] Experimento completado. μ={resultados['acuracia_media']:.4f}")
        return jsonify(respuesta)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/diagnosticar", methods=["POST"])
def diagnosticar():
    if gerenciador is None:
        return jsonify({"status": "error", "message": "Ejecute /analizar primero"}), 400

    try:
        img_np = None
        if "image" in request.files:
            img = Image.open(request.files["image"]).convert("RGB")
            img_np = np.array(img)
        elif request.json and "image_base64" in request.json:
            img_data = base64.b64decode(request.json["image_base64"])
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img_np = np.array(img)
        else:
            return jsonify({"status": "error", "message": "Enviar imagen en multipart/form-data (campo 'image') o base64"}), 400

        preprocessado = pipeline_completo_preprocessamento(
            np.expand_dims(img_np, axis=0), tamanho_alvo=(64, 64),
        )
        caracteristicas, _ = extrair_todas_caracteristicas(
            preprocessado["imagens_cinza"],
            preprocessado["mascaras_binarias"],
            incluir_hog=gerenciador.incluir_hog,
            escalonador=gerenciador.escalonador,
        )

        memoria = gerenciador.memoria
        distancias, indices_ordenados = memoria.obter_distancias(caracteristicas)
        dist_ordenadas = distancias[0, indices_ordenados[0]]
        classe_predita = memoria.classificar(caracteristicas)[0]
        nomes = memoria.obter_mapeamento_classes_para_nomes()

        top3 = []
        for rank in range(min(3, len(dist_ordenadas))):
            idx = indices_ordenados[0, rank]
            rotulo = memoria.mapeamento_indice_para_classe[idx]
            clave = nomes.get(int(rotulo), f"Classe {rotulo}")
            top3.append({"rank": rank + 1, "clase": clave, "distancia": float(dist_ordenadas[rank])})

        clave_predita = nomes.get(int(classe_predita), f"Classe {classe_predita}")
        confianza = 1.0 / (1.0 + float(dist_ordenadas[0]))

        return jsonify({
            "status": "ok",
            "diagnostico": clave_predita,
            "confianza": round(confianza, 4),
            "top3": top3,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/resultados", methods=["GET"])
def obtener_resultados():
    if resultados is None:
        return jsonify({"status": "error", "message": "Ejecute /analizar primero"}), 400
    return jsonify({
        "status": "ok",
        "config": config_actual,
        "resultados": _serializar_resultados(),
        "memoria": _serializar_memoria(),
        "escalonador": _serializar_escalonador(),
        "total_classes_vistas": gerenciador.total_classes_vistas if gerenciador else None,
    })


# ---------------------------------------------------------------------------
# Inicio
# ---------------------------------------------------------------------------

def _obtener_url_colab(port):
    """Intenta obtener URL pública vía ngrok o proxy de Colab."""
    # Opción 1: ngrok con token
    import os as _os
    token = _os.environ.get("NGROK_AUTH_TOKEN")
    if token:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(token)
            url = ngrok.connect(port)
            print(f"  ngrok URL: {url}")
            return str(url)
        except Exception as e:
            print(f"  ngrok falló: {e}")

    # Opción 2: Proxy nativo de Colab (no requiere registro)
    try:
        from google.colab.output import eval_js
        url = eval_js(f"google.colab.kernel.proxyPort({port})")
        print(f"  Colab proxy URL: {url}")
        return url
    except Exception:
        pass

    # Opción 3: ngrok sin token (solo si ya está autenticado en el sistema)
    try:
        from pyngrok import ngrok
        url = ngrok.connect(port)
        print(f"  ngrok URL: {url}")
        return str(url)
    except Exception as e:
        print(f"  ngrok (sin token) falló: {e}")

    return None


def iniciar_servidor(host="0.0.0.0", port=5000, data_dir="data/raw", authtoken=None):
    """Llama a esta función desde una celda de Colab para arrancar el servidor.

    Para ngrok (opcional):
        desde https://dashboard.ngrok.com/get-started/your-authtoken
        y pásalo como authtoken="tu_token" o setea la variable NGROK_AUTH_TOKEN.
    """
    global DATA_DIR
    DATA_DIR = data_dir

    if authtoken:
        import os as _os
        _os.environ["NGROK_AUTH_TOKEN"] = authtoken

    public_url = _obtener_url_colab(port)

    print(f"\n{'='*60}")
    print(f"  🚀 API de Colab activa!")
    if public_url:
        print(f"  URL pública: {public_url}")
    else:
        print(f"  ⚠ Sin URL pública (solo accesible desde el notebook)")
        print(f"  Servidor local en http://{host}:{port}")
    print(f"  Endpoints:")
    print(f"    GET  /health            - health check")
    print(f"    POST /iniciar           - cargar dataset + sesión base (incremental)")
    print(f"    POST /ejecutar-sesion   - ejecutar 1 sesión incremental")
    print(f"    GET  /estado            - estado actual del experimento")
    print(f"    POST /analizar          - ejecutar experimento completo")
    print(f"    POST /diagnosticar      - clasificar imagen")
    print(f"    GET  /resultados        - obtener resultados")
    print(f"{'='*60}\n")

    app.run(host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Servidor FSCIL para Colab")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--ngrok-token", default=None, help="Token de autenticación ngrok (opcional, gratis en ngrok.com)")
    args = parser.parse_args()
    iniciar_servidor(host=args.host, port=args.port, data_dir=args.data_dir, authtoken=args.ngrok_token)
