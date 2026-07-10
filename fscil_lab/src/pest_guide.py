"""Guía de plagas y enfermedades foliares del dataset PlantVillage.

Proporciona información estructurada sobre cada clase: nombre común,
cultivo afectado, tipo (enfermedad/plaga/sano), agente causal,
síntomas y tratamiento sugerido.
"""

GUIA_COMPLETA = {
    "Apple___Apple_scab": {
        "cultivo": "Manzano",
        "nombre_comun": "Sarna del manzano",
        "tipo": "enfermedad",
        "agente_causal": "Venturia inaequalis (hongo)",
        "sintomas": "Manchas verde oliva a marrón oscuro en hojas y frutos, deformación y caída prematura de hojas.",
        "tratamiento": "Fungicidas a base de cobre o azufre, poda de ramas infectadas, variedades resistentes.",
    },
    "Apple___Black_rot": {
        "cultivo": "Manzano",
        "nombre_comun": "Podredumbre negra",
        "tipo": "enfermedad",
        "agente_causal": "Botryosphaeria obtusa (hongo)",
        "sintomas": "Manchas púrpuras en hojas que se expanden, pudrición negra en frutos, cancros en ramas.",
        "tratamiento": "Eliminación de tejido infectado, fungicidas protectantes, buena circulación de aire.",
    },
    "Apple___Cedar_apple_rust": {
        "cultivo": "Manzano",
        "nombre_comun": "Roya del enebro y manzano",
        "tipo": "enfermedad",
        "agente_causal": "Gymnosporangium juniperi-virginianae (hongo)",
        "sintomas": "Manchas anaranjadas brillantes en hojas, lesiones en frutos, requiere enebro como huésped alterno.",
        "tratamiento": "Fungicidas en primavera, eliminación de enebros cercanos, variedades resistentes.",
    },
    "Apple___healthy": {
        "cultivo": "Manzano",
        "nombre_comun": "Manzano sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hoja verde uniforme, sin manchas, deformaciones ni decoloraciones.",
        "tratamiento": "Mantenimiento preventivo: riego adecuado, fertilización balanceada.",
    },
    "Blueberry___healthy": {
        "cultivo": "Arándano",
        "nombre_comun": "Arándano sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hoja verde brillante, borde ligeramente aserrado, sin lesiones.",
        "tratamiento": "Suelo ácido (pH 4.5-5.5), riego constante, poda de formación.",
    },
    "Cherry___Powdery_mildew": {
        "cultivo": "Cerezo",
        "nombre_comun": "Mildiu polvoriento del cerezo",
        "tipo": "enfermedad",
        "agente_causal": "Podosphaera clandestina (hongo)",
        "sintomas": "Polvo blanco en hojas jóvenes, enrollamiento y distorsión foliar, reducción de fotosíntesis.",
        "tratamiento": "Azufre micronizado, bicarbonato de potasio, fungicidas sistémicos.",
    },
    "Cherry___healthy": {
        "cultivo": "Cerezo",
        "nombre_comun": "Cerezo sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hojas verdes lanceoladas, márgenes finamente aserrados, textura lisa.",
        "tratamiento": "Poda de mantenimiento, control de humedad, fertilización orgánica.",
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "cultivo": "Maíz",
        "nombre_comun": "Mancha gris de la hoja (Cercospora)",
        "tipo": "enfermedad",
        "agente_causal": "Cercospora zeae-maydis (hongo)",
        "sintomas": "Lesiones rectangulares de color gris a marrón claro paralelas a las venas foliares.",
        "tratamiento": "Rotación de cultivos, fungicidas foliares, híbridos resistentes, labranza mínima.",
    },
    "Corn___Common_rust": {
        "cultivo": "Maíz",
        "nombre_comun": "Roya común del maíz",
        "tipo": "enfermedad",
        "agente_causal": "Puccinia sorghi (hongo)",
        "sintomas": "Pústulas ovaladas de color marrón rojizo en ambas caras de la hoja, liberan esporas polvorientas.",
        "tratamiento": "Fungicidas a base de triazol o estrobilurina, variedades resistentes, siembra temprana.",
    },
    "Corn___Northern_Leaf_Blight": {
        "cultivo": "Maíz",
        "nombre_comun": "Tizón foliar del norte",
        "tipo": "enfermedad",
        "agente_causal": "Exserohilum turcicum (hongo)",
        "sintomas": "Lesiones alargadas de color gris verdoso a marrón claro, forma de cigarro, coalescen en áreas necróticas.",
        "tratamiento": "Híbridos resistentes, rotación de cultivos, fungicidas protectantes, eliminación de residuos.",
    },
    "Corn___healthy": {
        "cultivo": "Maíz",
        "nombre_comun": "Maíz sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hojas largas y anchas, verde intenso, venas paralelas bien definidas, sin lesiones.",
        "tratamiento": "Fertilización nitrogenada, riego por surcos, control de malezas.",
    },
    "Grape___Black_rot": {
        "cultivo": "Uva",
        "nombre_comun": "Podredumbre negra de la vid",
        "tipo": "enfermedad",
        "agente_causal": "Guignardia bidwellii (hongo)",
        "sintomas": "Manchas marrón claro con borde oscuro en hojas, frutos momificados de color negro.",
        "tratamiento": "Fungicidas protectantes (mancozeb), eliminación de racimos infectados, poda de aireación.",
    },
    "Grape___Esca_(Black_Measles)": {
        "cultivo": "Uva",
        "nombre_comun": "Esca o sarampión negro",
        "tipo": "enfermedad",
        "agente_causal": "Phaeoacremonium aleophilum, Phaeomoniella chlamydospora (hongos)",
        "sintomas": "Estrías cloróticas entre venas, manchas necróticas, decoloración púrpura, hojas 'tigre'.",
        "tratamiento": "Poda de madera muerta, protección de heridas de poda, eliminación de cepas afectadas.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cultivo": "Uva",
        "nombre_comun": "Tizón foliar (mancha de Isariopsis)",
        "tipo": "enfermedad",
        "agente_causal": "Isariopsis clavispora (hongo)",
        "sintomas": "Manchas pequeñas marrón rojizo con márgenes oscuros, coalescen en áreas necróticas grandes.",
        "tratamiento": "Fungicidas cúpricos, poda de aireación, eliminación de hojas infectadas del suelo.",
    },
    "Grape___healthy": {
        "cultivo": "Uva",
        "nombre_comun": "Vid sana",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hoja palmada verde intenso, lóbulos bien definidos, superficie brillante, sin manchas.",
        "tratamiento": "Conducción en espaldera, poda verde, riego por goteo, fertilización potásica.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cultivo": "Naranjo",
        "nombre_comun": "Huanglongbing (enverdecimiento de los cítricos)",
        "tipo": "enfermedad",
        "agente_causal": "Candidatus Liberibacter spp. (bacteria transmitida por psílidos)",
        "sintomas": "Manchas amarillas asimétricas en hojas, frutos pequeños deformes y amargos, muerte regresiva.",
        "tratamiento": "No tiene cura. Control del psílido vector (Diaphorina citri), eliminación de árboles infectados.",
    },
    "Peach___Bacterial_spot": {
        "cultivo": "Durazno",
        "nombre_comun": "Mancha bacteriana del duraznero",
        "tipo": "enfermedad",
        "agente_causal": "Xanthomonas campestris pv. pruni (bacteria)",
        "sintomas": "Manchas acuosas que se tornan marrón púrpura, centros necróticos que se desprenden (disparo de escopeta).",
        "tratamiento": "Fungicidas cúpricos en otoño, variedades resistentes, evitar riego por aspersión.",
    },
    "Peach___healthy": {
        "cultivo": "Durazno",
        "nombre_comun": "Duraznero sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hojas lanceoladas verde oscuro, bordes finamente serrados, textura ligeramente coriácea.",
        "tratamiento": "Poda de formación, fertilización NPK, control de monilia, aclareo de frutos.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "cultivo": "Pimiento morrón",
        "nombre_comun": "Mancha bacteriana del pimiento",
        "tipo": "enfermedad",
        "agente_causal": "Xanthomonas campestris pv. vesicatoria (bacteria)",
        "sintomas": "Manchas marrón oscuro elevadas en hojas, defoliación, lesiones en frutos.",
        "tratamiento": "Semillas certificadas libres de bacteria, fungicidas cúpricos, rotación de cultivos.",
    },
    "Pepper,_bell___healthy": {
        "cultivo": "Pimiento morrón",
        "nombre_comun": "Pimiento sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hoja verde brillante en forma de lanza, textura lisa, márgenes enteros.",
        "tratamiento": "Riego constante, mulch plástico, fertilización equilibrada, tutoreo.",
    },
    "Potato___Early_blight": {
        "cultivo": "Papa",
        "nombre_comun": "Tizón temprano",
        "tipo": "enfermedad",
        "agente_causal": "Alternaria solani (hongo)",
        "sintomas": "Manchas marrón oscuro con anillos concéntricos en hojas inferiores, amarillamiento y defoliación.",
        "tratamiento": "Fungicidas protectantes, rotación de cultivos (evitar solanáceas), riego por surcos.",
    },
    "Potato___Late_blight": {
        "cultivo": "Papa",
        "nombre_comun": "Tizón tardío",
        "tipo": "enfermedad",
        "agente_causal": "Phytophthora infestans (oomiceto)",
        "sintomas": "Manchas verde grisáceas acuosas que se expanden rápido, margen clorótico, esporulación blanca en envés.",
        "tratamiento": "Fungicidas sistémicos (metalaxil), destrucción de residuos, variedades resistentes, alerta temprana.",
    },
    "Potato___healthy": {
        "cultivo": "Papa",
        "nombre_comun": "Papa sana",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hojas compuestas verde oscuro, folíolos ovalados, superficie ligeramente pubescente, sin manchas.",
        "tratamiento": "Aporque, fertilización, riego por surcos, control de insectos vectores.",
    },
    "Raspberry___healthy": {
        "cultivo": "Frambuesa",
        "nombre_comun": "Frambueso sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hojas compuestas verde oscuro, envés blanquecino, bordes dentados, sin lesiones visibles.",
        "tratamiento": "Poda de cañas viejas, tutoreo, riego por goteo, fertilización orgánica.",
    },
    "Soybean___healthy": {
        "cultivo": "Soja",
        "nombre_comun": "Soja sana",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hojas trifoliadas verde intenso, forma ovalada, pubescencia fina, venación reticulada.",
        "tratamiento": "Fijación biológica de nitrógeno, control de malezas, rotación de cultivos.",
    },
    "Squash___Powdery_mildew": {
        "cultivo": "Calabaza",
        "nombre_comun": "Mildiu polvoriento de las cucurbitáceas",
        "tipo": "enfermedad",
        "agente_causal": "Podosphaera xanthii (hongo)",
        "sintomas": "Polvo blanco talcoso en haz y envés de hojas, amarillamiento progresivo, reducción de rendimiento.",
        "tratamiento": "Azufre mojable, bicarbonato de sodio, fungicidas sistémicos, variedades tolerantes.",
    },
    "Strawberry___Leaf_scorch": {
        "cultivo": "Fresa",
        "nombre_comun": "Quemadura de la hoja de la fresa",
        "tipo": "enfermedad",
        "agente_causal": "Diplocarpon earlianum (hongo)",
        "sintomas": "Manchas púrpura oscuro a rojizo que coalescen en los bordes, dando aspecto de quemadura.",
        "tratamiento": "Fungicidas protectantes, eliminación de hojas viejas, acolchado para evitar salpicaduras.",
    },
    "Strawberry___healthy": {
        "cultivo": "Fresa",
        "nombre_comun": "Fresa sana",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hojas trifoliadas verde brillante, pecíolos largos, márgenes dentados, superficie glabra.",
        "tratamiento": "Acolchado plástico, riego por goteo, fertilización, renovación de plantación cada 2-3 años.",
    },
    "Tomato___Bacterial_spot": {
        "cultivo": "Tomate",
        "nombre_comun": "Mancha bacteriana del tomate",
        "tipo": "enfermedad",
        "agente_causal": "Xanthomonas vesicatoria (bacteria)",
        "sintomas": "Manchas marrón oscuro a negro en hojas y frutos, bordes acuosos, defoliación.",
        "tratamiento": "Semillas certificadas, fungicidas cúpricos, rotación de solanáceas, evitar aspersión.",
    },
    "Tomato___Early_blight": {
        "cultivo": "Tomate",
        "nombre_comun": "Tizón temprano del tomate",
        "tipo": "enfermedad",
        "agente_causal": "Alternaria solani (hongo)",
        "sintomas": "Manchas con anillos concéntricos en hojas inferiores, amarillamiento, cancros en tallo.",
        "tratamiento": "Fungicidas preventivos (clorotalonil), mulching, poda de hojas bajas, rotación.",
    },
    "Tomato___Late_blight": {
        "cultivo": "Tomate",
        "nombre_comun": "Tizón tardío del tomate",
        "tipo": "enfermedad",
        "agente_causal": "Phytophthora infestans (oomiceto)",
        "sintomas": "Manchas verde grisáceas irregulares, esporulación blanca en bordes, pudrición de frutos.",
        "tratamiento": "Fungicidas sistémicos, eliminación de plantas infectadas, evitar exceso de humedad.",
    },
    "Tomato___Leaf_Mold": {
        "cultivo": "Tomate",
        "nombre_comun": "Moho de la hoja del tomate",
        "tipo": "enfermedad",
        "agente_causal": "Passalora fulva (hongo)",
        "sintomas": "Manchas verde pálido a amarillo en haz, moho aterciopelado oliva en envés, defoliación.",
        "tratamiento": "Reducir humedad ambiental, fungicidas protectantes, ventilación en invernadero.",
    },
    "Tomato___Septoria_leaf_spot": {
        "cultivo": "Tomate",
        "nombre_comun": "Mancha foliar por Septoria",
        "tipo": "enfermedad",
        "agente_causal": "Septoria lycopersici (hongo)",
        "sintomas": "Pequeñas manchas circulares con borde oscuro y centro gris claro, puntos negros (picnidios).",
        "tratamiento": "Fungicidas protectantes, eliminación de hojas infectadas, rotación, mulching.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cultivo": "Tomate",
        "nombre_comun": "Araña roja (ácaro de dos manchas)",
        "tipo": "plaga",
        "agente_causal": "Tetranychus urticae (ácaro)",
        "sintomas": "Punteado amarillo en hojas, telarañas finas en envés, bronceado y caída de hojas.",
        "tratamiento": "Acaricidas, depredadores naturales (Phytoseiulus persimilis), mantener humedad alta.",
    },
    "Tomato___Target_Spot": {
        "cultivo": "Tomate",
        "nombre_comun": "Mancha anillada (Target spot)",
        "tipo": "enfermedad",
        "agente_causal": "Corynespora cassiicola (hongo)",
        "sintomas": "Manchas marrón oscuro con anillos concéntricos en hojas, frutos y tallos.",
        "tratamiento": "Fungicidas de amplio espectro, rotación de cultivos, semillas libres de patógeno.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cultivo": "Tomate",
        "nombre_comun": "Virus del enrollado amarillo de la hoja del tomate",
        "tipo": "enfermedad",
        "agente_causal": "Tomato yellow leaf curl virus (TYLCV, virus transmitido por mosca blanca)",
        "sintomas": "Enrollamiento de hojas hacia arriba, amarillamiento de bordes, reducción del crecimiento.",
        "tratamiento": "Control de mosca blanca (Bemisia tabaci), mallas anti-insectos, variedades resistentes.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "cultivo": "Tomate",
        "nombre_comun": "Virus del mosaico del tomate (ToMV)",
        "tipo": "enfermedad",
        "agente_causal": "Tomato mosaic virus (virus mecánico)",
        "sintomas": "Mosaico verde claro-oscuro en hojas, deformación foliar, moteado en frutos.",
        "tratamiento": "Semillas tratadas, desinfección de herramientas, rotación, variedades resistentes.",
    },
    "Tomato___healthy": {
        "cultivo": "Tomate",
        "nombre_comun": "Tomate sano",
        "tipo": "sano",
        "agente_causal": "Ninguno",
        "sintomas": "Hoja verde oscuro, compuesta, folíolos ovalados con bordes ligeramente dentados, textura suave.",
        "tratamiento": "Fertilización equilibrada, poda de formación, riego por goteo, entutorado.",
    },
}


def obtener_info_clave(clave: str) -> dict:
    """Retorna la información de una plaga/enfermedad por su clave PlantVillage."""
    return GUIA_COMPLETA.get(clave, {})


def obtener_nombre_comun(clave: str) -> str:
    """Retorna el nombre común de la plaga/enfermedad a partir de la clave del dataset."""
    info = GUIA_COMPLETA.get(clave, {})
    if info:
        return f"{info['nombre_comun']} ({info['cultivo']})"
    return clave


def obtener_cultivo(clave: str) -> str:
    """Extrae el nombre del cultivo de la clave del dataset (parte anterior a ___)."""
    return clave.split("___")[0] if "___" in clave else clave


def listar_por_cultivo() -> dict:
    """Agrupa las enfermedades/plagas por cultivo."""
    por_cultivo = {}
    for clave, info in GUIA_COMPLETA.items():
        cultivo = info["cultivo"]
        if cultivo not in por_cultivo:
            por_cultivo[cultivo] = []
        por_cultivo[cultivo].append({"clave": clave, **info})
    return dict(sorted(por_cultivo.items()))


def listar_por_tipo() -> dict:
    """Agrupa por tipo: enfermedad, plaga, sano."""
    por_tipo = {"enfermedad": [], "plaga": [], "sano": []}
    for clave, info in GUIA_COMPLETA.items():
        tipo = info.get("tipo", "enfermedad")
        if tipo in por_tipo:
            por_tipo[tipo].append({"clave": clave, **info})
    return por_tipo


def obtener_estadisticas() -> dict:
    """Retorna estadísticas de la guía."""
    total = len(GUIA_COMPLETA)
    cultivos = len(set(info["cultivo"] for info in GUIA_COMPLETA.values()))
    por_tipo = listar_por_tipo()
    return {
        "total_clases": total,
        "total_cultivos": cultivos,
        "enfermedades": len(por_tipo["enfermedad"]),
        "plagas": len(por_tipo["plaga"]),
        "sanos": len(por_tipo["sano"]),
    }
