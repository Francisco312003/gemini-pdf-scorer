import google.generativeai as genai
import os
import json


def cargar_api_key():
    """
    Carga la clave API desde un archivo gemini_key.json en el Escritorio.
    """
    try:
        ruta_escritorio = os.path.join(os.path.expanduser('~'), 'Desktop')
        ruta_json = os.path.join(ruta_escritorio, 'gemini_key.json')

        with open(ruta_json, 'r') as f:
            config = json.load(f)
            return config['api_key']
    except Exception as e:
        print(f"Error fatal al leer 'gemini_key.json': {e}")
        return None


# --- Configurar la API ---
API_KEY = cargar_api_key()
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("No se pudo cargar la API Key. Saliendo.")
    exit()

# --- La Función Principal: Listar Modelos ---
print("--- Buscando modelos disponibles para tu API Key... ---")

try:
    for m in genai.list_models():
        # Imprimir el nombre del modelo
        print(f"\nModelo: {m.name}")

        # Imprimir los métodos que soporta (¡esto es lo que nos interesa!)
        print(f"  Soporta 'generateContent': {'generateContent' in m.supported_generation_methods}")
        print(f"  Métodos Soportados: {m.supported_generation_methods}")

except Exception as e:
    print(f"\n--- ¡ERROR! ---")
    print(f"Ocurrió un error al intentar listar los modelos: {e}")
    print("Esto podría deberse a un problema con la API key o la conexión.")

print("\n--- Fin de la lista ---")