# Extractor Semántico de Opiniones de Productos

Este proyecto tiene como objetivo construir un sistema avanzado que procese reseñas de productos en línea, extraiga eventos y opiniones, relacione productos, usuarios y sentimientos, y genere representaciones estructuradas para búsqueda, análisis y visualización semántica.

---

## Objetivo

Desarrollar una plataforma que transforme reseñas en bruto en conocimiento navegable y consultable, facilitando:
- **Extracción de eventos y opiniones**
- **Detección de relaciones producto-usuario-sentimiento**
- **Generación de grafos y tripletas RDF**
- **Búsqueda y visualización semántica de experiencias**

---

## Componentes Principales

1. **Expresiones Regulares**  
   Identificar menciones de precios, fechas y modelos en textos libres.

2. **NER (Reconocimiento de Entidades Nombradas)**  
   Extraer marcas, productos, ubicaciones, fechas, personas y valores.

3. **Relationship Extraction**  
   Determinar relaciones explícitas e implícitas tales como (usuario, compró, producto).

4. **Event Extraction**  
   Identificar eventos clave: compras, devoluciones, quejas, recomendaciones.

5. **Modelo de Recuperación de Opiniones**  
   Permitir búsqueda eficiente de reseñas similares por producto, experiencia o sentimiento (BM25 o modelos de embeddings).

6. **Representación del Conocimiento**  
   Generar tripletas semánticas del tipo (entidad, relación, entidad/sentimiento).

7. **Web Semántica / Grafo RDF**  
   Convertir relaciones a grafos RDF consultables y visualizables.

---

## Etapas del Proyecto

1. **Dataset**
   - Recolectar reseñas de Amazon, Google Play u otras tiendas online.
   - Almacenar como CSV estructurado.

2. **Limpieza + Regex**
   - Eliminar ruido y normalizar textos.
   - Extraer patrones de fechas de compra, precios, modelos.

3. **NER**
   - Detectar entidades relevantes:
     - Producto → "Smartphone Z12"
     - Marca → "Sony"
     - Lugar → "México"
     - Persona → "Usuario123"

4. **Extracción de eventos**
   - Detectar verbos clave: "compré", "devolví", "probé", "funcionó", "falló".
   - Generar eventos tipo:
     - (Usuario, compró, Producto)
     - (Producto, falló en, Fecha)

5. **Extracción de relaciones**
   - Ejemplos:
     - (Carlos, recomendó, Refrigerador FríoTech)
     - (Sony, lanzó, Z12)

6. **Representación del conocimiento**
   - Generar tripletas RDF:
     - (Usuario123, compró, Lavadora X300)
     - (Lavadora X300, tiene_sentimiento, negativo)

7. **Web semántica y búsqueda**
   - Crear grafo RDF consultable:
     - Por producto, marca, ubicación, tipo de evento, sentimiento, etc.
   - Ranking de reseñas similares (ej: BM25).
   - Visualización de relaciones y emociones.

---

## Resultados Esperados

- **Grafo semántico navegable** de experiencias y opiniones de usuarios sobre productos.
- **Búsqueda semántica avanzada**, por ejemplo:
  > "Mostrar productos con más quejas sobre batería en México."
- **Visualización interactiva** de relaciones entre productos, marcas, usuarios y emociones.
- **Insights accionables** para fabricantes, consumidores y analistas.

---

## Requisitos Sugeridos

- **Python 3.x**
- [pandas](https://pandas.pydata.org/) (procesamiento de datos)
- [spaCy](https://spacy.io/) o [transformers](https://huggingface.co/transformers/) (NER y NLP)
- [rdflib](https://rdflib.readthedocs.io/) (RDF y grafos)
- [networkx](https://networkx.org/) o [graphviz](https://graphviz.gitlab.io/) (visualización)
- [scikit-learn](https://scikit-learn.org/) (BM25 u otros modelos de recuperación)
- [huggingface-datasets](https://huggingface.co/docs/datasets/) (descarga de datos)

---

## Ejemplo de Extracción

| Entidad   | Ejemplo extraído      |
|-----------|----------------------|
| Producto  | Smartphone Z12       |
| Marca     | Sony                 |
| Lugar     | México               |
| Persona   | Usuario123           |
| Precio    | $1,299.00            |
| Fecha     | 12/05/2023           |
| Modelo    | Z12                  |
| Evento    | Usuario123 compró Smartphone Z12 en 12/05/2023 |
| Sentimiento | Negativo           |

---

## Créditos y Referencias

- Dataset: [Amazon Reviews 2023 - McAuley-Lab](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Inspirado en tareas de procesamiento de lenguaje natural, web semántica y análisis de opiniones.

---

¡Contribuciones, ideas y sugerencias son bienvenidas!
