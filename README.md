# 📄 Gemini pdf scorer: AI-Powered PDF Evaluator

**Gemini pdf scorer** es una herramienta automatizada que analiza, puntúa y clasifica documentos académicos (PDFs) utilizando la inteligencia artificial de **Google Gemini**.

Diseñado para investigadores y desarrolladores que necesitan procesar grandes volúmenes de literatura científica y extraer insights estructurados en formato JSON.

## 🚀 Características Principales

* **Análisis Inteligente:** Utiliza `Gemini 1.5 Flash` para leer y entender el contexto de cada PDF.
* **Puntuación Estructurada:** Evalúa relevancia, metodología y claridad del 1 al 10.
* **Salida JSON:** Genera datos limpios y listos para ser consumidos por otras aplicaciones o dashboards.
* **Procesamiento por Lotes:** Analiza carpetas enteras de documentos automáticamente.

## 🛠️ Stack Tecnológico

* **Python 3.10+**
* **Google Generative AI (Gemini API)**
* **Pandas** (Procesamiento de datos)
* **PyPDF** (Extracción de texto)

## 📦 Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/Francisco312003/gemini-pdf-scorer.git](https://github.com/Francisco312003/gemini-pdf-scorer.git)
    cd gemini-pdf-scorer
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurar Variables de Entorno:**
    Crea un archivo `.env` en la raíz del proyecto y agrega tu API Key de Google:
    ```env
    GOOGLE_API_KEY=tu_clave_aqui_sin_comillas
    ```

## 💻 Uso

1.  Coloca tus archivos PDF en la carpeta `pdfs_to_analyze/`.
2.  Ejecuta el script principal:
    ```bash
    python main.py
    ```
3.  Revisa los resultados en `analysis_report.json` o `analysis_report.csv`.

## 📊 Ejemplo de Salida (JSON)

```json
{
    "filename": "paper_2024.pdf",
    "title": "Advanced Neural Networks in Medical Imaging",
    "relevance_score": 9,
    "key_findings": [
        "Improved accuracy by 15% using new attention mechanism",
        "Reduced training time by half"
    ],
    "recommendation": "Accept"
}
Desarrollado por Francisco Padilla
