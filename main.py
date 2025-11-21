# =============================================================================
# SLR AUTOMÁTICA HÍBRIDA - Versión Mejorada (Noviembre 2025)
# Autor:
# Stack: Gemini 1.5 Pro + Semantic Scholar API + pdfplumber + pandas
# Objetivo: Puntuar ~400 PDFs según rúbrica PC1-PC5 y guardar en Excel
# =============================================================================

import time
import json
import logging
import pdfplumber
import pandas as pd
import google.generativeai as genai
import requests
from urllib.parse import quote_plus
from pathlib import Path
from rapidfuzz import process, fuzz
import joblib
import re
from typing import Dict, Any, Tuple, Optional
from google.api_core import exceptions as google_exceptions

# ----------------- CONFIGURACIÓN DE LOGGING -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("slr_processing.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------- CARGA DE API KEYS -----------------
try:
    with open(Path("~/Desktop/gemini_key.json").expanduser(), "r") as f:
        API_KEY = json.load(f)["api_key"]
    genai.configure(api_key=API_KEY)
    logger.info("Clave de Gemini cargada correctamente")
except Exception as e:
    logger.error(f"No se pudo cargar la API key: {e}")
    exit(1)

# --- DEBUG: Listar modelos disponibles ---
logger.info("--- Buscando modelos disponibles para tu API Key... ---")
for m in genai.list_models():
    supports_generate = 'generateContent' in m.supported_generation_methods
    logger.info(f"Modelo: {m.name}")
    logger.info(f"  Soporta 'generateContent': {supports_generate}")
    logger.info(f"  Métodos Soportados: {m.supported_generation_methods}")
logger.info("--- Fin de la lista ---")

# ----------------- CARGA DE BASE DE DATOS SCIMAGO -----------------
SCIMAGO_PATH = Path("scimago_rankings.csv")
if not SCIMAGO_PATH.exists():
    logger.error(f"No se encontró {SCIMAGO_PATH}")
    exit(1)

df_scimago = pd.read_csv(SCIMAGO_PATH, sep=';', on_bad_lines='skip')
df_scimago['Title_lower'] = df_scimago['Title'].str.lower().str.strip()
logger.info(f"Scimago cargado: {len(df_scimago)} registros")

# ----------------- CARGA DE LINKS DESDE OTRO EXCEL -----------------
LINKS_PATH = Path(r"C:\Users\richa\Desktop\Proyecto_SLR\LINK SLR.xlsx")
if LINKS_PATH.exists():
    df_links = pd.read_excel(LINKS_PATH)
    logger.info(f"Columnas en {LINKS_PATH}: {df_links.columns.tolist()}")
    if len(df_links.columns) > 21:
        df_links['Título_lower'] = df_links.iloc[:, 5].astype(str).str.lower().str.strip()  # Col F index 5
        logger.info(f"Links cargados desde {LINKS_PATH}: {len(df_links)} registros. Usando col F para Título.")
    else:
        df_links = None
        logger.error("Excel tiene menos columnas de las esperadas. Links no se agregarán.")
else:
    df_links = None
    logger.warning(f"No se encontró {LINKS_PATH}. Links no se agregarán.")

# ----------------- CACHE SEMANTIC SCHOLAR -----------------
CACHE_FILE = Path("semantic_scholar_cache.pkl")
if CACHE_FILE.exists():
    semantic_cache = joblib.load(CACHE_FILE)
else:
    semantic_cache = {}
logger.info(f"Cache de Semantic Scholar: {len(semantic_cache)} entradas previas")

# ----------------- PROMPT PARA GEMINI (MODIFICADO PARA CITAS Y TEXTOS) -----------------
def crear_prompt_gemini(texto_pdf: str) -> str:
    return f"""
    Eres un experto en revisión sistemática de literatura.
    Analiza el artículo y devuelve SOLO un JSON válido con este esquema exacto:

    {{
      "extraccion": {{
        "titulo_articulo": "string",
        "venue_nombre": "string",
        "venue_tipo": "journal|conference|unknown",
        "ano_publicacion": "YYYY" or null
      }},
      "evaluacion": [
        {{"id": "PC3", "puntaje": 0 or 0.5 or 1, "justificacion": "máx 150 caracteres, cita partes específicas del texto (e.g., sección X, párrafo Y)"}},
        {{"id": "PC4", "puntaje": 0 or 0.5 or 1, "justificacion": "máx 150 caracteres, cita partes específicas del texto (e.g., sección X, párrafo Y)"}},
        {{"id": "PC5", "puntaje": 0 or 0.5 or 1, "justificacion": "máx 150 caracteres, cita partes específicas del texto (e.g., sección X, párrafo Y)"}}
      ],
      "textos": {{
        "objetivo_texto": "extracto literal del objetivo (máx 200 chars)",
        "motivacion_texto": "extracto literal de la motivación (máx 200 chars)",
        "resultados_texto": "extracto literal de los resultados (máx 200 chars)"
      }}
    }}

    Texto del artículo:
    {texto_pdf}
    """

# ----------------- LLAMADA A GEMINI -----------------
def evaluar_con_ia(texto_pdf: str) -> Optional[Dict[str, Any]]:
    prompt = crear_prompt_gemini(texto_pdf[:200_000])

    MODELOS_TUYOS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-001",
        "gemini-flash-latest"
    ]

    for modelo in MODELOS_TUYOS:
        try:
            logger.info(f"Probando modelo: {modelo}")
            model = genai.GenerativeModel(
                modelo,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=8192
                )
            )
            response = model.generate_content(prompt)

            raw_text = response.candidates[0].content.parts[0].text
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if not json_match:
                logger.error("Gemini no devolvió JSON válido")
                continue

            datos = json.loads(json_match.group(0))
            logger.info(f"ÉXITO CON {modelo} - JSON recibido correctamente")
            return datos

        except google_exceptions.ResourceExhausted as e:
            wait = 70
            match = re.search(r'retry in ([\d\.]+)s', str(e), re.IGNORECASE)
            if match:
                wait = int(float(match.group(1))) + 10
            logger.warning(f"Cuota agotada ({modelo}). Esperando {wait} segundos...")
            time.sleep(wait)
            continue

        except Exception as e:
            logger.warning(f"Error con {modelo}: {e}. Probando siguiente...")
            continue

    logger.error("Todos los modelos fallaron.")
    return None

# ----------------- PC1 -----------------
def obtener_score_pc1(venue_nombre: str, venue_tipo: str) -> Tuple[float, str]:
    if not venue_nombre or venue_nombre.strip() == "":
        return 0.0, "Sin venue detectado"

    venue_tipo = venue_tipo.lower().strip()

    if venue_tipo == "conference":
        return 0.0, "Conferencia → revisar CORE manualmente"

    if venue_tipo not in ["journal", "unknown"]:
        return 0.0, f"Tipo raro: {venue_tipo}"

    matches = process.extractOne(
        venue_nombre.lower().strip(),
        df_scimago['Title_lower'],
        scorer=fuzz.token_sort_ratio
    )

    if matches and matches[1] >= 80:
        row = df_scimago[df_scimago['Title_lower'] == matches[0]].iloc[0]
        quartile = row.get('SJR Best Quartile', row.get('Quartile', 'Unknown'))
        score = 1.0 if quartile in ['Q1', 'Q2'] else 0.5 if quartile in ['Q3', 'Q4'] else 0.0
        return score, f"Scimago {quartile} ({matches[1]}% match)"
    else:
        return 0.0, f"No encontrado en Scimago ({matches[1] if matches else 0}% match)"

# ----------------- PC2 -----------------
def obtener_score_pc2(titulo_articulo: str, ano_publicacion: str) -> Tuple[float, str, int, int]:
    if not titulo_articulo:
        return 0.0, "Falta título", 0, 0

    cache_key = titulo_articulo.lower().strip()
    if cache_key in semantic_cache:
        citas, año_api = semantic_cache[cache_key]
    else:
        query = quote_plus(titulo_articulo.strip())
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": 1, "fields": "title,citationCount,year"}
        headers = {"User-Agent": "SLR-Automator/1.0 (tuemail@dominio.com)"}

        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 429:
                logger.warning("Rate limit Semantic Scholar. Pausa 60s")
                time.sleep(60)
                return obtener_score_pc2(titulo_articulo, ano_publicacion)
            r.raise_for_status()
            data = r.json()

            if data.get("total", 0) > 0 and data.get("data"):
                paper = data["data"][0]
                citas = paper.get("citationCount", 0)
                año_api = paper.get("year")
            else:
                citas, año_api = 0, None

            semantic_cache[cache_key] = (citas, año_api)
            joblib.dump(semantic_cache, CACHE_FILE)

        except Exception as e:
            logger.warning(f"Error Semantic Scholar: {e}")
            return 0.0, f"Error API: {e}", 0, 0

    try:
        ano = año_api or (int(ano_publicacion) if ano_publicacion and ano_publicacion.isdigit() else 2025)
    except:
        ano = 2025

    if ano <= 2017:
        score = 1.0 if citas > 5 else 0.5 if citas >= 1 else 0.0
    else:
        score = 1.0 if citas > 0 else 0.0

    return score, f"Semantic Scholar: {citas} citas (año {ano})", citas, ano

# ----------------- PROCESAMIENTO PRINCIPAL -----------------
def procesar_pdfs_y_guardar_excel(
        carpeta_pdfs: str = "test_pdfs",
        archivo_salida: str = "resultados_formato_CALIDAD.xlsx"
):
    carpeta = Path(carpeta_pdfs)
    pdf_files = list(carpeta.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No se encontraron PDFs en {carpeta}")
        return

    logger.info(f"Iniciando procesamiento de {len(pdf_files)} PDFs en '{carpeta}'...")

    filas_calidad = []
    conferencias_pendientes = []

    for idx, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"--- Procesando [{idx}/{len(pdf_files)}] {pdf_path.name} ---")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                texto = "\n".join(page.extract_text() or "" for page in pdf.pages[:20])
            logger.info(f"Texto extraído: {len(texto):,} caracteres")
        except Exception as e:
            logger.error(f"Error al leer PDF: {e}")
            continue

        datos_ia = evaluar_con_ia(texto)
        if not datos_ia:
            logger.error(f"Fallo de Gemini en {pdf_path.name}")
            fila = [pdf_path.name, "", "", "Fallo de todos los modelos Gemini", "", "", 0, 0, 2025, 0, "", "", 0, "", 0, 0, "", "", "", 0, 0, "", "", "", 0, 0, "", "", "", 0, 0, ""]
            filas_calidad.append(fila)
            continue

        ext = datos_ia.get("extraccion", {})
        eva = {item["id"]: item for item in datos_ia.get("evaluacion", [])}
        textos = datos_ia.get("textos", {})

        pc1_score, pc1_just = obtener_score_pc1(ext.get("venue_nombre", ""), ext.get("venue_tipo", "unknown"))
        pc2_score, pc2_just, citas, ano = obtener_score_pc2(ext.get("titulo_articulo", ""), ext.get("ano_publicacion", None))

        if ext.get("venue_tipo", "").lower() == "conference":
            conferencias_pendientes.append({
                "archivo": pdf_path.name,
                "titulo": ext.get("titulo_articulo", ""),
                "venue": ext.get("venue_nombre", ""),
                "año": ano
            })

        fila = ["" for _ in range(32)]

        fila[0] = pdf_path.name
        titulo = ext.get("titulo_articulo", "")
        fila[1] = titulo

        # Link desde otro Excel (fuzzy match, umbral bajado a 70%, logging para debug)
        link = ""
        if df_links is not None:
            matches = process.extractOne(
                titulo.lower().strip(),
                df_links['Título_lower'],
                scorer=fuzz.token_sort_ratio
            )
            if matches:
                logger.info(f"Match score para '{titulo}': {matches[1]}%")
            if matches and matches[1] >= 70:  # Bajado para más matches
                row_link = df_links[df_links['Título_lower'] == matches[0]].iloc[0]
                link = row_link.iloc[21]  # Col V index 21
                logger.info(f"Link encontrado para '{titulo}': {link}")
            else:
                logger.warning(f"No match suficiente para título: '{titulo}' (score: {matches[1] if matches else 0})")
        fila[2] = link

        pc3_data = eva.get("PC3", {"puntaje": 0, "justificacion": "Fallo IA"})
        pc4_data = eva.get("PC4", {"puntaje": 0, "justificacion": "Fallo IA"})
        pc5_data = eva.get("PC5", {"puntaje": 0, "justificacion": "Fallo IA"})
        observacion = f"PC1: {pc1_just}; PC2: {pc2_just}; PC3: {pc3_data['justificacion']}; PC4: {pc4_data['justificacion']}; PC5: {pc5_data['justificacion']}"
        fila[3] = observacion[:500]

        # PC1: Poner score en la casilla correspondiente
        if pc1_score == 1.0:
            fila[4] = pc1_score
        elif pc1_score == 0.5:
            fila[5] = pc1_score
        else:
            fila[6] = pc1_score
        fila[7] = pc1_score  # Puntuación PC1

        fila[8] = ano
        fila[9] = citas

        # PC2: Poner score en la casilla correspondiente
        if ano <= 2017:
            if citas > 5:
                fila[10] = pc2_score
            elif citas >= 1:
                fila[11] = pc2_score
            else:
                fila[12] = pc2_score
        else:
            if citas > 0:
                fila[13] = pc2_score
            else:
                fila[14] = pc2_score
        fila[15] = pc2_score  # Puntuación PC2

        fila[16] = textos.get("objetivo_texto", "")  # Texto literal en Q
        pc3_score = pc3_data["puntaje"]
        if pc3_score == 1.0:
            fila[17] = pc3_score
        elif pc3_score == 0.5:
            fila[18] = pc3_score
        else:
            fila[19] = pc3_score
        fila[20] = pc3_score

        fila[21] = textos.get("motivacion_texto", "")  # Texto literal en V
        pc4_score = pc4_data["puntaje"]
        if pc4_score == 1.0:
            fila[22] = pc4_score
        elif pc4_score == 0.5:
            fila[23] = pc4_score
        else:
            fila[24] = pc4_score
        fila[25] = pc4_score

        fila[26] = textos.get("resultados_texto", "")  # Texto literal en AA
        pc5_score = pc5_data["puntaje"]
        if pc5_score == 1.0:
            fila[27] = pc5_score
        elif pc5_score == 0.5:
            fila[28] = pc5_score
        else:
            fila[29] = pc5_score
        fila[30] = pc5_score

        fila[31] = ""  # Dejar vacío para fórmula en Excel

        filas_calidad.append(fila)

        if idx % 10 == 0 or idx == len(pdf_files):
            logger.info(f"Checkpoint guardado: {idx} artículos procesados")
            try:
                df_checkpoint = pd.DataFrame(filas_calidad)
                with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
                    df_checkpoint.to_excel(writer, sheet_name='Calidad', index=False, header=False)
            except Exception as e:
                logger.error(f"Error al guardar checkpoint de Excel: {e}")

            pd.DataFrame(conferencias_pendientes).to_csv("conferencias_pendientes.csv", index=False)

    logger.info(f"Guardando archivo Excel final en '{archivo_salida}'...")
    df_final_calidad = pd.DataFrame(filas_calidad)

    try:
        with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
            df_final_calidad.to_excel(writer, sheet_name='Calidad', index=False, header=False)
    except Exception as e:
        logger.error(f"Error fatal al guardar Excel final: {e}")

    pd.DataFrame(conferencias_pendientes).to_csv("conferencias_pendientes.csv", index=False)

    logger.info("¡PROCESO COMPLETADO!")
    logger.info(f"Resultados guardados en '{archivo_salida}' (Hoja: 'Calidad')")
    logger.info(f"Conferencias para revisión manual: {len(conferencias_pendientes)}")

# ----------------- EJECUTAR -----------------
if __name__ == "__main__":
    procesar_pdfs_y_guardar_excel(
        carpeta_pdfs="test_pdfs",
        archivo_salida="resultados_formato_CALIDAD.xlsx"
    )