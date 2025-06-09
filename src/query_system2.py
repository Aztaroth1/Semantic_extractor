import pandas as pd
import spacy
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from deep_translator import GoogleTranslator
import json
from collections import defaultdict, Counter

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords')

# Configuración de colores
COLOR_PRIMARY = "#3498db"
COLOR_SECONDARY = "#2c3e50"
COLOR_ACCENT = "#e74c3c"
COLOR_BACKGROUND = "#ecf0f1"
COLOR_TEXT = "#2c3e50"
COLOR_CARD = "#ffffff"
COLOR_STARS = "#f39c12"

class ReviewSearchApp:
    def __init__(self, root, data_path):
        self.root = root
        self.root.title("Sistema de Búsqueda Semántica de Reseñas")
        self.root.geometry("1200x900")
        self.root.configure(bg=COLOR_BACKGROUND)

        # Cargar sistema de consultas
        try:
            self.query_system = UniversalReviewQuerySystem(data_path)
            print("Sistema cargado exitosamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el sistema: {str(e)}")
            self.root.destroy()
            return

        # Variables para la búsqueda avanzada
        self.advanced_search_vars = {}

        # Cargar iconos
        self.load_icons()

        # Crear interfaz
        self.create_widgets()

    def load_icons(self):
        self.icons = {
            "search": "🔍",
            "star": "★",
            "exit": "🚪",
            "info": "ℹ️",
            "tech": "💻",
            "beauty": "💄",
            "music": "🎸",
            "graph": "📊",
            "semantic": "🧠",
            "filter": "🔧"
        }

    def create_widgets(self):
        # Notebook para pestañas
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pestaña 1: Búsqueda semántica
        self.create_semantic_search_tab(notebook)

        # Pestaña 2: Búsqueda avanzada
        self.create_advanced_search_tab(notebook)

        # Pestaña 3: Grafo semántico
        self.create_graph_tab(notebook)
        self.configure_text_tags()

    def create_semantic_search_tab(self, notebook):
        # Frame principal para búsqueda semántica
        search_frame = ttk.Frame(notebook)
        notebook.add(search_frame, text=f"{self.icons['semantic']} Búsqueda Semántica")

        # Cabecera
        header_frame = ttk.Frame(search_frame)
        header_frame.pack(fill=tk.X, pady=(10, 15))

        title_label = ttk.Label(
            header_frame,
            text="🌟 Búsqueda Semántica Inteligente 🌟",
            font=("Arial", 18, "bold"),
            foreground=COLOR_SECONDARY
        )
        title_label.pack(pady=10)

        # Ejemplos de consultas semánticas
        examples_frame = ttk.LabelFrame(search_frame, text="Ejemplos de Consultas Semánticas", padding=10)
        examples_frame.pack(fill=tk.X, padx=10, pady=5)

        examples = [
            "productos con quejas sobre batería en México",
            "experiencias negativas con pantallas",
            "reseñas positivas de Samsung en España",
            "problemas de durabilidad en electrónicos"
        ]

        for i, example in enumerate(examples):
            btn = ttk.Button(
                examples_frame,
                text=f"💡 {example}",
                command=lambda ex=example: self.set_search_query(ex)
            )
            btn.pack(side=tk.LEFT if i < 2 else tk.LEFT, padx=5, pady=2)
            if i == 1: # Nueva línea después de 2 botones
                ttk.Frame(examples_frame).pack()

        # Buscador principal
        search_main_frame = ttk.Frame(search_frame, padding=10)
        search_main_frame.pack(fill=tk.X, padx=10, pady=10)

        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(
            search_main_frame,
            textvariable=self.search_var,
            width=60,
            font=("Arial", 12)
        )
        search_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        search_entry.bind("<Return>", lambda event: self.perform_semantic_search())

        search_btn = ttk.Button(
            search_main_frame,
            text=f"{self.icons['search']} Buscar",
            command=self.perform_semantic_search,
            style="Accent.TButton"
        )
        search_btn.pack(side=tk.LEFT)

        # Resultados
        results_frame = ttk.LabelFrame(search_frame, text="Resultados", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            bg=COLOR_CARD,
            padx=10,
            pady=10
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Listo para búsqueda semántica")
        status_bar = ttk.Label(search_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

    def create_advanced_search_tab(self, notebook):
        # Frame para búsqueda avanzada
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text=f"{self.icons['filter']} Búsqueda Avanzada")

        # Título
        title_label = ttk.Label(
            advanced_frame,
            text="🔧 Búsqueda Avanzada por Filtros",
            font=("Arial", 16, "bold"),
            foreground=COLOR_SECONDARY
        )
        title_label.pack(pady=10)

        # Frame para filtros
        filters_frame = ttk.LabelFrame(advanced_frame, text="Filtros de Búsqueda", padding=15)
        filters_frame.pack(fill=tk.X, padx=10, pady=10)

        # Crear campos de filtro
        filter_fields = [
            ("Producto:", "product"),
            ("Marca:", "brand"),
            ("Sentimiento:", "sentiment"),
            ("Ubicación:", "location"),
            ("Palabra clave:", "keyword")
        ]

        for i, (label, key) in enumerate(filter_fields):
            row = i // 2
            col = i % 2

            ttk.Label(filters_frame, text=label, font=("Arial", 10, "bold")).grid(
                row=row, column=col*2, sticky="w", padx=(0, 5), pady=5
            )

            var = tk.StringVar()
            self.advanced_search_vars[key] = var
            entry = ttk.Entry(filters_frame, textvariable=var, width=25)
            entry.grid(row=row, column=col*2+1, sticky="ew", padx=(0, 20), pady=5)

        # Configurar expansión de columnas
        for i in range(4):
            filters_frame.columnconfigure(i, weight=1 if i % 2 == 1 else 0)

        # Botón de búsqueda avanzada
        search_advanced_btn = ttk.Button(
            filters_frame,
            text=f"{self.icons['search']} Buscar con Filtros",
            command=self.perform_advanced_search,
            style="Accent.TButton"
        )
        search_advanced_btn.grid(row=3, column=0, columnspan=4, pady=10)

        # Resultados avanzados
        self.advanced_results_frame = ttk.LabelFrame(advanced_frame, text="Resultados de Búsqueda Avanzada", padding=10)
        self.advanced_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.advanced_results_text = scrolledtext.ScrolledText(
            self.advanced_results_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            bg=COLOR_CARD
        )
        self.advanced_results_text.pack(fill=tk.BOTH, expand=True)

    def create_graph_tab(self, notebook):
        # Frame para visualización de grafos
        graph_frame = ttk.Frame(notebook)
        notebook.add(graph_frame, text=f"{self.icons['graph']} Grafo Semántico")

        # Título
        title_label = ttk.Label(
            graph_frame,
            text="📊 Visualización del Grafo Semántico",
            font=("Arial", 16, "bold"),
            foreground=COLOR_SECONDARY
        )
        title_label.pack(pady=10)

        # Controles del grafo
        controls_frame = ttk.Frame(graph_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            controls_frame,
            text="🔗 Generar Grafo Completo",
            command=self.show_complete_graph
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            controls_frame,
            text="😊 Grafo de Sentimientos",
            command=self.show_sentiment_graph
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            controls_frame,
            text="🏪 Grafo de Productos",
            command=self.show_product_graph
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            controls_frame,
            text="💾 Exportar RDF",
            command=self.export_rdf_graph_handler
        ).pack(side=tk.LEFT, padx=5)

        # Frame para el grafo
        self.graph_display_frame = ttk.Frame(graph_frame)
        self.graph_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Info del grafo
        self.graph_info_text = scrolledtext.ScrolledText(
            self.graph_display_frame,
            height=8,
            wrap=tk.WORD,
            font=("Arial", 9)
        )
        self.graph_info_text.pack(fill=tk.X, pady=(0, 10))

    def configure_text_tags(self):
        self.results_text.tag_configure("header", foreground=COLOR_PRIMARY, font=("Arial", 12, "bold"))
        self.results_text.tag_configure("subheader", foreground=COLOR_SECONDARY, font=("Arial", 10, "bold"))
        self.results_text.tag_configure("data", foreground=COLOR_TEXT, font=("Arial", 10))
        self.results_text.tag_configure("review", foreground="#34495e", font=("Arial", 10))
        self.results_text.tag_configure("stars", foreground=COLOR_STARS, font=("Arial", 12))
        self.results_text.tag_configure("triple", foreground="#8e44ad", font=("Arial", 9, "italic"))
        
        self.advanced_results_text.tag_configure("header", foreground=COLOR_PRIMARY, font=("Arial", 12, "bold"))
        self.advanced_results_text.tag_configure("subheader", foreground=COLOR_SECONDARY, font=("Arial", 10, "bold"))
        self.advanced_results_text.tag_configure("data", foreground=COLOR_TEXT, font=("Arial", 10))
        self.advanced_results_text.tag_configure("triple", foreground="#8e44ad", font=("Arial", 9, "italic"))


    def set_search_query(self, query):
        self.search_var.set(query)
        self.perform_semantic_search()

    def perform_semantic_search(self):
        query = self.search_var.get().strip()
        if not query:
            self.status_var.set("Ingrese una consulta")
            return

        self.status_var.set(f"Procesando consulta semántica: '{query}'...")
        self.root.update_idletasks()

        try:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            # Primero intentar búsqueda semántica RDF
            rdf_results = self.query_system.semantic_rdf_query(query)

            # Luego búsqueda semántica tradicional
            semantic_results = self.query_system.enhanced_semantic_search(query)

            if not semantic_results and not rdf_results:
                self.results_text.insert(tk.END, "No se encontraron resultados relevantes.", "data")
                self.status_var.set(f"0 resultados para: '{query}'")
            else:
                total_results = len(semantic_results)
                self.results_text.insert(tk.END, f"🎯 {total_results} resultados semánticos para: '{query}'\n\n", "header")

                # Mostrar triples RDF relacionados
                if rdf_results:
                    self.results_text.insert(tk.END, "🔗 Relaciones semánticas encontradas:\n", "subheader")
                    for triple in rdf_results[:10]: # Limitar a 10
                        self.results_text.insert(tk.END, f"    • {triple[0]} → {triple[1]} → {triple[2]}\n", "triple")
                    self.results_text.insert(tk.END, "\n")

                # Mostrar resultados detallados
                for i, res in enumerate(semantic_results[:10]): # Limitar a 10 resultados
                    self.display_result(res, i+1)

                self.status_var.set(f"{total_results} resultados semánticos encontrados")

        except Exception as e:
            self.results_text.insert(tk.END, f"Error en búsqueda semántica: {str(e)}", "data")
            self.status_var.set(f"Error: {str(e)}")

        finally:
            self.results_text.config(state=tk.DISABLED)

    def perform_advanced_search(self):
        # Obtener valores de filtros
        filters = {key: var.get().strip() for key, var in self.advanced_search_vars.items()}
        filters = {k: v for k, v in filters.items() if v} # Solo filtros no vacíos

        if not filters:
            messagebox.showwarning("Filtros vacíos", "Ingrese al menos un filtro para la búsqueda")
            return

        try:
            self.advanced_results_text.config(state=tk.NORMAL)
            self.advanced_results_text.delete(1.0, tk.END)

            # Realizar búsqueda avanzada
            results = self.query_system.advanced_semantic_search(
                product=filters.get('product'),
                brand=filters.get('brand'),
                sentiment=filters.get('sentiment'),
                location=filters.get('location'),
                failure_keyword=filters.get('keyword'),
                top_n=15
            )

            if not results:
                self.advanced_results_text.insert(tk.END, "No se encontraron resultados con estos filtros.", "data")
            else:
                self.advanced_results_text.insert(tk.END, f"🔍 {len(results)} resultados con filtros aplicados:\n\n", "header")

                for i, res in enumerate(results):
                    self.display_advanced_result(res, i+1)

        except Exception as e:
            self.advanced_results_text.insert(tk.END, f"Error en búsqueda avanzada: {str(e)}", "data")

        finally:
            self.advanced_results_text.config(state=tk.DISABLED)

    def display_result(self, res, num):
        self.results_text.insert(tk.END, f"\n🎯 Resultado #{num} ", "header")
        self.results_text.insert(tk.END, f"(Relevancia: {res['score']})\n", "subheader")

        # Mostrar triples RDF si existen
        if 'triples' in res and res['triples']:
            self.results_text.insert(tk.END, "🔗 Relaciones:\n", "subheader")
            for triple in res['triples']:
                self.results_text.insert(tk.END, f"    {triple[0]} → {triple[1]} → {triple[2]}\n", "triple")
            self.results_text.insert(tk.END, "\n")

        # Mostrar datos estructurados
        for key, value in res['data'].items():
            if value and value != 'N/A':
                if key == 'Rating':
                    self.results_text.insert(tk.END, f"{key}: ", "subheader")
                    self.results_text.insert(tk.END, f"{value}\n", "stars")
                else:
                    self.results_text.insert(tk.END, f"{key}: ", "subheader")
                    self.results_text.insert(tk.END, f"{value}\n", "data")

        # Mostrar texto de reseña
        if res.get('text'):
            self.results_text.insert(tk.END, "\n📝 Reseña:\n", "subheader")
            preview = (res['text'][:300] + '...') if len(res['text']) > 300 else res['text']
            self.results_text.insert(tk.END, f"{preview}\n", "review")

        self.results_text.insert(tk.END, "\n" + "="*80 + "\n")

    def display_advanced_result(self, res, num):
        text_widget = self.advanced_results_text
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, f"\n📊 Resultado #{num} ", "header")
        text_widget.insert(tk.END, f"(Score: {res.get('score', 'N/A')})\n", "subheader")

        # Mostrar triples si existen
        if 'triples' in res and res['triples']:
            text_widget.insert(tk.END, "🔗 Relaciones semánticas:\n", "subheader")
            for triple in res['triples']:
                text_widget.insert(tk.END, f"    • {triple[0]} → {triple[1]} → {triple[2]}\n", "triple")
            text_widget.insert(tk.END, "\n")

        # Mostrar datos del resultado
        data = res.get('data', {})
        for key, value in data.items():
            if value and value != 'N/A':
                text_widget.insert(tk.END, f"{key}: {value}\n", "data")

        text_widget.insert(tk.END, "\n" + "-"*60 + "\n")
        text_widget.config(state=tk.DISABLED)

    def show_complete_graph(self):
        try:
            graph_info = self.query_system.visualize_complete_semantic_graph()
            self.update_graph_info(graph_info)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el grafo: {str(e)}")

    def show_sentiment_graph(self):
        try:
            graph_info = self.query_system.visualize_sentiment_network()
            self.update_graph_info(graph_info)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el grafo de sentimientos: {str(e)}")

    def show_product_graph(self):
        try:
            graph_info = self.query_system.visualize_product_network()
            self.update_graph_info(graph_info)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el grafo de productos: {str(e)}")

    def update_graph_info(self, info):
        self.graph_info_text.config(state=tk.NORMAL)
        self.graph_info_text.delete(1.0, tk.END)
        self.graph_info_text.insert(tk.END, info)
        self.graph_info_text.config(state=tk.DISABLED)

    def export_rdf_graph_handler(self):
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")],
                title="Exportar Grafo RDF como..."
            )
            if not filename:
                return

            export_path = self.query_system.export_rdf_triples_csv(filename)
            messagebox.showinfo("Exportación exitosa", f"Grafo RDF exportado como:\n{export_path}")

        except Exception as e:
            messagebox.showerror("Error de exportación", f"No se pudo exportar el grafo RDF:\n{str(e)}")

class UniversalReviewQuerySystem:
    def __init__(self, data_path):
        print("Inicializando sistema de consultas...")

        # Cargar datos y procesar
        self.df = pd.read_csv(data_path).fillna('')

        if self.df.empty:
            raise ValueError("El dataset está vacío")

        print(f"Dataset cargado: {len(self.df)} reseñas")

        # Cargar modelo NLP
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("Modelo spaCy cargado exitosamente")
        except:
            print("⚠️ Modelo spaCy no encontrado. Funcionalidad NLP limitada.")
            self.nlp = None

        # Preprocesar textos
        self.texts = self.df['text'].astype(str).tolist()
        self.tokenized_texts = self._preprocess_texts()

        if self.tokenized_texts:
            self.bm25 = BM25Okapi(self.tokenized_texts)
            print("Índice BM25 construido")
        else:
            raise ValueError("No hay textos válidos para la búsqueda")

        # Construir grafo RDF
        self.build_enhanced_rdf_graph()
        print("Grafo RDF construido")

        # Extraer entidades y sentimientos
        self._extract_semantic_features()
        print("Características semánticas extraídas")

        print("Sistema inicializado correctamente ✅")

    def _preprocess_texts(self):
        """Tokeniza y limpia los textos para BM25"""
        tokenized = []
        english_stopwords = set(stopwords.words('english'))

        for text in self.texts:
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens
                      if t not in english_stopwords
                      and len(t) > 2
                      and not t.isdigit()
                      and re.match(r'^[a-zA-Z]+$', t)]

            if tokens:
                tokenized.append(tokens)

        return tokenized

    def _extract_semantic_features(self):
        """Extrae características semánticas de las reseñas"""
        self.semantic_features = {
            'products': defaultdict(list),
            'brands': defaultdict(list),
            'locations': defaultdict(list),
            'sentiments': defaultdict(list),
            'problems': defaultdict(list)
        }

        # Palabras clave para problemas comunes
        problem_keywords = {
            'battery': ['battery', 'batería', 'duración', 'carga'],
            'screen': ['screen', 'pantalla', 'display', 'brightness'],
            'durability': ['break', 'broken', 'fragile', 'durability'],
            'performance': ['slow', 'lag', 'performance', 'speed']
        }

        for idx, row in self.df.iterrows():
            text = str(row.get('text', ''))

            # Extraer entidades nombradas existentes
            product = str(row.get('ner_products', ''))
            brand = str(row.get('ner_brands', ''))
            location = str(row.get('ner_locations', ''))

            if product and product != 'nan':
                self.semantic_features['products'][product].append(idx)
            if brand and brand != 'nan':
                self.semantic_features['brands'][brand].append(idx)
            if location and location != 'nan':
                self.semantic_features['locations'][location].append(idx)

            # Detectar problemas en el texto
            text_lower = text.lower()
            for problem_type, keywords in problem_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    self.semantic_features['problems'][problem_type].append(idx)

    def build_enhanced_rdf_graph(self):
        """Construye un grafo RDF mejorado con más relaciones semánticas"""
        self.rdf_graph = defaultdict(set)
        self.entity_relations = defaultdict(lambda: defaultdict(set))

        for idx, row in self.df.iterrows():
            # Entidades básicas
            product = str(row.get('ner_products', '')).strip()
            brand = str(row.get('ner_brands', '')).strip()
            location = str(row.get('ner_locations', '')).strip()
            person = str(row.get('ner_persons', '')).strip()

            text = str(row.get('text', ''))

            # Determinar sentimiento básico
            sentiment = self._analyze_sentiment(text)

            # Crear triples RDF
            if product and product not in ['', 'nan', 'N/A']:
                if brand and brand not in ['', 'nan', 'N/A']:
                    self.rdf_graph[(product, 'es_de_marca')].add(brand)
                    self.rdf_graph[(brand, 'fabrica')].add(product)

                if location and location not in ['', 'nan', 'N/A']:
                    self.rdf_graph[(product, 'vendido_en')].add(location)
                    self.rdf_graph[(location, 'vende')].add(product)

                if person and person not in ['', 'nan', 'N/A']:
                    self.rdf_graph[(person, 'compró')].add(product)
                    self.rdf_graph[(product, 'comprado_por')].add(person)

                # Relaciones de sentimiento
                self.rdf_graph[(product, 'tiene_sentimiento')].add(sentiment)
                self.rdf_graph[(sentiment, 'asociado_con')].add(product)

                # Problemas detectados
                problems = self._detect_problems(text)
                for problem in problems:
                    self.rdf_graph[(product, 'tiene_problema')].add(problem)
                    self.rdf_graph[(problem, 'afecta_a')].add(product)

    def _analyze_sentiment(self, text):
        """Análisis básico de sentimiento"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'problem']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return 'positivo'
        elif neg_count > pos_count:
            return 'negativo'
        else:
            return 'neutro'

    def _detect_problems(self, text):
        """Detecta problemas mencionados en el texto"""
        problems = []
        text_lower = text.lower()

        problem_patterns = {
            'batería': ['battery', 'batería', 'charge', 'power'],
            'pantalla': ['screen', 'display', 'pantalla'],
            'durabilidad': ['break', 'broken', 'crack', 'fragile'],
            'rendimiento': ['slow', 'lag', 'performance', 'freeze']
        }

        for problem, keywords in problem_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                problems.append(problem)

        return problems

    def semantic_rdf_query(self, query_text):
        """Consulta semántica del grafo RDF"""
        query_lower = query_text.lower()
        results = []

        # Patrones de consulta semántica
        if any(word in query_lower for word in ['quejas', 'problemas', 'negativo', 'malo']):
            # Buscar productos con sentimiento negativo
            negative_products = self.rdf_graph.get(('negativo', 'asociado_con'), set())
            for product in negative_products:
                results.extend(self.query_rdf_graph(subject=product, predicate='tiene_problema'))

        # Buscar por ubicación específica
        for location in ['méxico', 'mexico', 'españa', 'spain']:
            if location in query_lower:
                location_products = self.rdf_graph.get((location.title(), 'vende'), set())
                for product in location_products:
                    results.extend(self.query_rdf_graph(subject=product))

        # Buscar por problemas específicos
        problem_map = {
            'batería': 'batería',
            'battery': 'batería',
            'pantalla': 'pantalla',
            'screen': 'pantalla',
            'durabilidad': 'durabilidad'
        }

        for keyword, problem in problem_map.items():
            if keyword in query_lower:
                problem_products = self.rdf_graph.get((problem, 'afecta_a'), set())
                for product in problem_products:
                    results.extend(self.query_rdf_graph(subject=product, predicate='tiene_problema', obj=problem))

        # Si no hay resultados específicos, devolver muestra general
        if not results:
            all_triples = []
            for key, values in list(self.rdf_graph.items())[:50]: # Limitar para performance
                for value in values:
                    all_triples.append((key[0], key[1], value))
            return all_triples[:20]

        return list(set(results))[:20] # Eliminar duplicados y limitar

    def query_rdf_graph(self, subject=None, predicate=None, obj=None):
        """Consulta el grafo RDF con patrones de triple"""
        results = []

        for (subj, pred), objects in self.rdf_graph.items():
            if (subject is None or subject == subj) and (predicate is None or predicate == pred):
                for o in objects:
                    if obj is None or obj == o:
                        results.append((subj, pred, o))

        return results

    def enhanced_semantic_search(self, query, top_n=10):
        """Búsqueda semántica mejorada con análisis de intención"""
        if not hasattr(self, 'bm25') or self.bm25 is None:
            return []

        # Análisis de intención de la consulta
        intent = self._analyze_query_intent(query)

        # Expandir consulta con sinónimos y términos relacionados
        expanded_query = self._expand_query(query)

        # Tokenizar consulta expandida
        tokens = word_tokenize(expanded_query.lower())
        english_stopwords = set(stopwords.words('english'))
        tokens = [t for t in tokens
                  if t not in english_stopwords
                  and len(t) > 2
                  and re.match(r'^[a-zA-Z]+$', t)] # Corregido aquí

        # Obtener puntuaciones BM25
        scores = self.bm25.get_scores(tokens)

        # Aplicar boost basado en intención
        boosted_scores = self._apply_intent_boost(scores, intent)

        # Obtener mejores resultados
        top_indices = np.argsort(boosted_scores)[::-1][:top_n*2]

        # Filtrar por relevancia mínima y diversidad
        min_score = 1.5
        results = []
        seen_products = set()

        for idx in top_indices:
            if boosted_scores[idx] > min_score:
                result = self._format_enhanced_result(idx, boosted_scores[idx], intent)

                # Evitar productos duplicados para diversidad
                product = result['data'].get('Producto', '')
                if product not in seen_products or len(results) < 3:
                    results.append(result)
                    if product:
                        seen_products.add(product)

                    if len(results) >= top_n:
                        break

        return results

    def _analyze_query_intent(self, query):
        """Analiza la intención de la consulta"""
        query_lower = query.lower()

        intent = {
            'sentiment': 'neutral',
            'problem_focus': False,
            'location_focus': False,
            'brand_focus': False,
            'comparison': False
        }

        # Detectar sentimiento
        if any(word in query_lower for word in ['quejas', 'problemas', 'malo', 'negativo', 'complaints', 'problems', 'bad']):
            intent['sentiment'] = 'negative'
        elif any(word in query_lower for word in ['bueno', 'positivo', 'recomendado', 'good', 'positive', 'recommended']):
            intent['sentiment'] = 'positive'

        # Detectar enfoque en problemas
        if any(word in query_lower for word in ['batería', 'battery', 'pantalla', 'screen', 'durabilidad', 'performance']):
            intent['problem_focus'] = True

        # Detectar enfoque geográfico
        if any(word in query_lower for word in ['méxico', 'mexico', 'españa', 'spain', 'en', 'ubicación']):
            intent['location_focus'] = True

        # Detectar enfoque en marcas
        if any(word in query_lower for word in ['samsung', 'apple', 'sony', 'marca', 'brand']):
            intent['brand_focus'] = True

        return intent

    def _expand_query(self, query):
        """Expande la consulta con sinónimos y términos relacionados"""
        query_lower = query.lower()

        # Diccionario de expansiones
        expansions = {
            'batería': 'battery power charge duración energía',
            'battery': 'batería power charge duration energy',
            'pantalla': 'screen display monitor visualización',
            'screen': 'pantalla display monitor visualization',
            'problems': 'problemas issues defects fallas',
            'problemas': 'problems issues defects failures',
            'méxico': 'mexico mexican latinoamerica',
            'mexico': 'méxico mexican latin america'
        }

        expanded = query
        for term, synonyms in expansions.items():
            if term in query_lower:
                expanded += ' ' + synonyms

        return expanded

    def _apply_intent_boost(self, scores, intent):
        """Aplica boost a las puntuaciones basado en la intención"""
        boosted_scores = scores.copy()

        for idx, score in enumerate(scores):
            row = self.df.iloc[idx]
            boost = 1.0

            # Boost por sentimiento
            text_lower = str(row.get('text', '')).lower()
            if intent['sentiment'] == 'negative':
                if any(word in text_lower for word in ['bad', 'terrible', 'problem', 'issue']):
                    boost *= 1.5
            elif intent['sentiment'] == 'positive':
                if any(word in text_lower for word in ['good', 'great', 'excellent', 'amazing']):
                    boost *= 1.5

            # Boost por problemas específicos
            if intent['problem_focus']:
                if any(word in text_lower for word in ['battery', 'screen', 'break', 'slow']):
                    boost *= 1.3

            # Boost por ubicación
            if intent['location_focus']:
                location = str(row.get('ner_locations', '')).lower()
                if location and any(loc in location for loc in ['mexico', 'méxico', 'spain', 'españa']):
                    boost *= 1.4

            boosted_scores[idx] = score * boost

        return boosted_scores

    def _format_enhanced_result(self, idx, score, intent):
        """Formatea resultado mejorado con información semántica"""
        row = self.df.iloc[idx]

        # Usar formato base
        base_result = self._format_result(idx, score)

        # Agregar información semántica
        text = str(row.get('text', ''))
        product = str(row.get('ner_products', ''))

        # Extraer triples RDF relevantes
        triples = []
        if product and product != 'nan':
            # Buscar triples relacionados con este producto
            for (subj, pred), objs in self.rdf_graph.items():
                if subj == product:
                    for obj in list(objs)[:3]: # Limitar a 3 por predicado
                        triples.append((subj, pred, obj))

        # Detectar problemas mencionados específicamente
        detected_problems = self._detect_problems(text)
        if detected_problems:
            for problem in detected_problems:
                triples.append((product if product != 'nan' else 'Producto', 'menciona_problema', problem))

        # Agregar información de intención
        base_result['intent_match'] = intent
        base_result['triples'] = triples
        base_result['problems_detected'] = detected_problems

        return base_result

    def advanced_semantic_search(self, product=None, brand=None, sentiment=None, location=None, failure_keyword=None, top_n=15):
        """Búsqueda semántica avanzada con filtros múltiples"""
        df = self.df.copy()

        # Aplicar filtros
        if product:
            df = df[df['ner_products'].str.contains(product, case=False, na=False)]

        if brand:
            df = df[df['ner_brands'].str.contains(brand, case=False, na=False)]

        if location:
            df = df[df['ner_locations'].str.contains(location, case=False, na=False)]

        if sentiment:
            # Filtrar por sentimiento en el texto
            if sentiment.lower() in ['negativo', 'negative']:
                mask = df['text'].str.contains('|'.join(['bad', 'terrible', 'awful', 'problem', 'issue']), case=False, na=False)
                df = df[mask]
            elif sentiment.lower() in ['positivo', 'positive']:
                mask = df['text'].str.contains('|'.join(['good', 'great', 'excellent', 'amazing', 'perfect']), case=False, na=False)
                df = df[mask]

        if df.empty:
            return []

        # Ranking por palabra clave de falla
        if failure_keyword:
            text_list = df['text'].astype(str).tolist()
            tokenized_texts = []

            for text in text_list:
                tokens = word_tokenize(text.lower())
                english_stopwords = set(stopwords.words('english'))
                tokens = [t for t in tokens if t not in english_stopwords and len(t) > 2]
                tokenized_texts.append(tokens)

            if tokenized_texts:
                bm25 = BM25Okapi(tokenized_texts)
                keyword_tokens = word_tokenize(failure_keyword.lower())
                keyword_tokens = [t for t in keyword_tokens if len(t) > 2]
                scores = bm25.get_scores(keyword_tokens)

                # Ordenar por relevancia
                top_indices = np.argsort(scores)[::-1][:top_n]
                selected_rows = df.iloc[top_indices]
            else:
                selected_rows = df.head(top_n)
        else:
            selected_rows = df.head(top_n)

        # Formatear resultados
        results = []
        for idx, (original_idx, row) in enumerate(selected_rows.iterrows()):
            score = scores[top_indices[idx]] if failure_keyword and 'scores' in locals() and top_indices.any() else 1.0


            # Extraer triples RDF
            triples = []
            product_name = str(row.get('ner_products', ''))
            brand_name = str(row.get('ner_brands', ''))
            location_name = str(row.get('ner_locations', ''))

            if product_name and product_name != 'nan':
                if brand_name and brand_name != 'nan':
                    triples.append((product_name, 'es_de_marca', brand_name))
                if location_name and location_name != 'nan':
                    triples.append((product_name, 'vendido_en', location_name))

                # Sentimiento detectado
                text_sentiment = self._analyze_sentiment(str(row.get('text', '')))
                triples.append((product_name, 'tiene_sentimiento', text_sentiment))

                # Problemas detectados
                problems = self._detect_problems(str(row.get('text', '')))
                for problem in problems:
                    triples.append((product_name, 'tiene_problema', problem))

            # Formatear resultado
            result = self._format_result(original_idx, score)
            result['triples'] = triples
            results.append(result)

        return results

    def visualize_complete_semantic_graph(self):
        """Visualiza el grafo semántico completo"""
        if not hasattr(self, 'rdf_graph'):
            self.build_enhanced_rdf_graph()

        G = nx.DiGraph()

        # Agregar nodos y aristas desde el grafo RDF
        edge_count = 0
        for (subj, pred), objs in self.rdf_graph.items():
            for obj in objs:
                if edge_count < 100: # Limitar para visualización
                    G.add_edge(subj, obj, relation=pred)
                    edge_count += 1

        # Crear visualización
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1.5, iterations=50)

        # Dibujar nodos por tipo
        product_nodes = [n for n in G.nodes() if any(n in products for products in self.semantic_features['products'].keys())]
        sentiment_nodes = [n for n in G.nodes() if n in ['positivo', 'negativo', 'neutro']]
        other_nodes = [n for n in G.nodes() if n not in product_nodes and n not in sentiment_nodes]

        nx.draw_networkx_nodes(G, pos, nodelist=product_nodes, node_color='lightblue', node_size=300, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=sentiment_nodes, node_color='lightcoral', node_size=400, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='lightgreen', node_size=200, alpha=0.8)

        # Dibujar aristas
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray', arrows=True, arrowsize=10)

        # Etiquetas selectivas
        important_nodes = product_nodes + sentiment_nodes
        labels = {n: n for n in important_nodes if len(n) < 15}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title("Grafo Semántico Completo - Productos, Sentimientos y Relaciones", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Información del grafo
        info = f"""
📊 INFORMACIÓN DEL GRAFO SEMÁNTICO COMPLETO

🔹 Nodos totales: {G.number_of_nodes()}
🔹 Aristas totales: {G.number_of_edges()}
🔹 Productos únicos: {len(product_nodes)}
🔹 Sentimientos: {len(sentiment_nodes)}
🔹 Otras entidades: {len(other_nodes)}

🔗 TIPOS DE RELACIONES:
• Productos ↔ Marcas: {len([e for e in G.edges(data=True) if e[2].get('relation') in ['es_de_marca', 'fabrica']])}
• Productos ↔ Sentimientos: {len([e for e in G.edges(data=True) if e[2].get('relation') == 'tiene_sentimiento'])}
• Productos ↔ Problemas: {len([e for e in G.edges(data=True) if e[2].get('relation') == 'tiene_problema'])}
• Productos ↔ Ubicaciones: {len([e for e in G.edges(data=True) if e[2].get('relation') in ['vendido_en', 'vende']])}

🎯 El grafo muestra las conexiones semánticas entre productos, marcas, sentimientos, 
   problemas y ubicaciones extraídas de las reseñas de usuarios.
        """

        return info

    def visualize_sentiment_network(self):
        """Visualiza la red de sentimientos por productos"""
        G = nx.Graph()

        # Crear red de productos y sentimientos
        sentiment_data = defaultdict(list)

        for idx, row in self.df.iterrows():
            product = str(row.get('ner_products', '')).strip()
            text = str(row.get('text', ''))

            if product and product not in ['', 'nan', 'N/A']:
                sentiment = self._analyze_sentiment(text)
                sentiment_data[product].append(sentiment)

        # Calcular sentimientos dominantes por producto
        for product, sentiments in sentiment_data.items():
            sentiment_counts = Counter(sentiments)
            dominant_sentiment = sentiment_counts.most_common(1)[0][0]
            total_reviews = len(sentiments)

            if total_reviews >= 2: # Solo productos con múltiples reseñas
                G.add_node(product, sentiment=dominant_sentiment, count=total_reviews)

        # Conectar productos con sentimientos similares
        products = list(G.nodes())
        for i, prod1 in enumerate(products):
            for prod2 in products[i+1:]:
                if G.nodes[prod1]['sentiment'] == G.nodes[prod2]['sentiment']:
                    G.add_edge(prod1, prod2, weight=0.5)

        # Visualización
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Colores por sentimiento
        color_map = {'positivo': 'lightgreen', 'negativo': 'lightcoral', 'neutro': 'lightyellow'}
        node_colors = [color_map.get(G.nodes[node]['sentiment'], 'lightgray') for node in G.nodes()]
        node_sizes = [G.nodes[node]['count'] * 50 for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

        # Etiquetas
        labels = {n: n[:15] + '...' if len(n) > 15 else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title("Red de Sentimientos por Productos", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Estadísticas
        sentiment_stats = Counter(G.nodes[node]['sentiment'] for node in G.nodes())

        info = f"""
😊 RED DE SENTIMIENTOS POR PRODUCTOS

📊 ESTADÍSTICAS:
🔹 Productos analizados: {len(G.nodes())}
🔹 Conexiones: {len(G.edges())}

💚 Productos con sentimiento POSITIVO: {sentiment_stats.get('positivo', 0)}
❤️ Productos con sentimiento NEGATIVO: {sentiment_stats.get('negativo', 0)}
💛 Productos con sentimiento NEUTRO: {sentiment_stats.get('neutro', 0)}

🔗 Los productos están conectados cuando comparten el mismo sentimiento dominante.
📏 El tamaño del nodo representa el número de reseñas analizadas.
        """

        return info

    def visualize_product_network(self):
        """Visualiza la red de productos por marcas y problemas"""
        G = nx.Graph()

        # Agregar productos y sus relaciones
        for idx, row in self.df.iterrows():
            product = str(row.get('ner_products', '')).strip()
            brand = str(row.get('ner_brands', '')).strip()
            text = str(row.get('text', ''))

            if product and product not in ['', 'nan', 'N/A']:
                # Detectar problemas
                problems = self._detect_problems(text)

                # Agregar nodo producto
                G.add_node(product, type='product')

                # Conectar con marca
                if brand and brand not in ['', 'nan', 'N/A']:
                    G.add_node(brand, type='brand')
                    G.add_edge(product, brand, relation='marca')

                # Conectar con problemas
                for problem in problems:
                    problem_node = f"Problema: {problem}"
                    G.add_node(problem_node, type='problem')
                    G.add_edge(product, problem_node, relation='problema')

        # Visualización
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Colores por tipo de nodo
        color_map = {'product': 'lightblue', 'brand': 'lightgreen', 'problem': 'lightcoral'}
        node_colors = [color_map.get(G.nodes.get(node, {}).get('type', 'unknown'), 'lightgray') for node in G.nodes()]

        # Tamaños de nodo
        node_sizes = []
        for node in G.nodes():
            if G.nodes[node].get('type') == 'product':
                node_sizes.append(400)
            elif G.nodes[node].get('type') == 'brand':
                node_sizes.append(300)
            else:
                node_sizes.append(200)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')

        # Etiquetas selectivas
        important_nodes = [n for n in G.nodes() if G.degree(n) > 1][:20]
        labels = {n: n[:12] + '...' if len(n) > 12 else n for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=7)

        plt.title("Red de Productos, Marcas y Problemas", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Análisis de centralidad
        centrality = nx.degree_centrality(G)
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

        info = f"""
🏪 RED DE PRODUCTOS, MARCAS Y PROBLEMAS

📊 ESTADÍSTICAS:
🔹 Nodos totales: {len(G.nodes())}
🔹 Conexiones: {len(G.edges())}
🔹 Productos: {len([n for n in G.nodes() if G.nodes[n].get('type') == 'product'])}
🔹 Marcas: {len([n for n in G.nodes() if G.nodes[n].get('type') == 'brand'])}
🔹 Problemas: {len([n for n in G.nodes() if G.nodes[n].get('type') == 'problem'])}

🎯 ENTIDADES MÁS CENTRALES:
"""

        for i, (entity, score) in enumerate(top_central[:5]):
            info += f"    {i+1}. {entity[:30]} (centralidad: {score:.3f})\n"

        info += """
🔗 La red muestra cómo los productos se relacionan con sus marcas y problemas reportados.
📏 El tamaño indica el tipo de entidad: productos (grande), marcas (medio), problemas (pequeño).
        """

        return info

    def export_rdf_triples_csv(self, filename):
        """Exporta las triples RDF a un archivo CSV"""
        triples_data = []

        # Convertir grafo RDF a lista de triples
        for (subject, predicate), objects in self.rdf_graph.items():
            for obj in objects:
                triples_data.append({
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj,
                    'type': 'semantic_relation'
                })

        # Agregar estadísticas del grafo
        stats = {
            'total_triples': len(triples_data),
            'unique_subjects': len(set(t['subject'] for t in triples_data)),
            'unique_predicates': len(set(t['predicate'] for t in triples_data)),
            'unique_objects': len(set(t['object'] for t in triples_data))
        }

        # Crear DataFrame y exportar
        df_triples = pd.DataFrame(triples_data)
        df_triples.to_csv(filename, index=False, encoding='utf-8')

        # Exportar también las estadísticas
        stats_filename = filename.replace('.csv', '_stats.json')
        with open(stats_filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✅ Grafo RDF exportado:")
        print(f"    📄 Triples: {filename}")
        print(f"    📊 Estadísticas: {stats_filename}")
        print(f"    🔢 Total triples: {stats['total_triples']}")

        return filename

    def _format_result(self, idx, score):
        """Formatea un resultado individual"""
        row = self.df.iloc[idx]

        get_val = lambda col: row[col] if col in row and pd.notna(row[col]) and row[col] != '' else 'N/A'

        producto = get_val('ner_products')
        marca = get_val('ner_brands')
        lugar = get_val('ner_locations')
        persona = get_val('ner_persons')
        precio = get_val('extracted_prices')
        fecha = get_val('extracted_purchase_dates')
        modelo = get_val('extracted_product_models')
        rating = get_val('rating')

        # Construir evento semántico
        evento = ""
        if persona != 'N/A' and producto != 'N/A' and fecha != 'N/A':
            evento = f"{persona} compró {producto} en {fecha}"
        elif persona != 'N/A' and producto != 'N/A':
            evento = f"{persona} compró {producto}"
        elif producto != 'N/A' and fecha != 'N/A':
            evento = f"Compra de {producto} en {fecha}"
        elif producto != 'N/A':
            evento = f"Experiencia con {producto}"
        else:
            evento = "Experiencia de usuario"

        # Formatear rating
        rating_stars = 'N/A'
        try:
            rating_val = float(rating)
            full_stars = int(rating_val)
            half_star = 1 if rating_val - full_stars >= 0.5 else 0
            empty_stars = 5 - full_stars - half_star

            rating_stars = '★' * full_stars
            if half_star:
                rating_stars += '½'
            rating_stars += '☆' * empty_stars
            rating_stars += f" ({rating_val}/5)"
        except (ValueError, TypeError):
            rating_stars = rating if rating != 'N/A' else 'Sin valoración'

        table_data = {
            'Producto': producto,
            'Marca': marca,
            'Ubicación': lugar,
            'Usuario': persona,
            'Precio': precio,
            'Fecha': fecha,
            'Modelo': modelo,
            'Rating': rating_stars,
            'Evento': evento
        }

        return {
            'review_id': idx,
            'score': round(score, 2),
            'data': table_data,
            'text': row['text'] if 'text' in row else ''
        }

if __name__ == "__main__":
    # Verificar existencia del archivo
    data_path = './Data/processed_data/Electronics_processed_ner.csv'
    if not os.path.exists(data_path):
        # Crear un root temporal para mostrar el messagebox si el archivo no existe
        root = tk.Tk()
        root.withdraw() # Ocultar la ventana principal
        messagebox.showerror("Error", f"Archivo no encontrado: {data_path}\nVerifique la ruta o ejecute primero el procesamiento de datos")
        root.destroy()
    else:
        print(f"🚀 Iniciando Sistema de Búsqueda Semántica...")
        print(f"📂 Cargando datos desde: {data_path}")

        root = tk.Tk()
        app = ReviewSearchApp(root, data_path)
        root.mainloop()