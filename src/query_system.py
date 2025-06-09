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

# Descargar recursos de NLTK
nltk.download('punkt')
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
        self.root.title("Sistema de Búsqueda de Reseñas")
        self.root.geometry("1100x800")
        self.root.configure(bg=COLOR_BACKGROUND)
        
        # Cargar sistema de consultas
        try:
            self.query_system = UniversalReviewQuerySystem(data_path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el sistema: {str(e)}")
            self.root.destroy()
            return
        
        # Cargar iconos
        self.load_icons()
        
        # Crear interfaz
        self.create_widgets()
    
    def load_icons(self):
        # Crear iconos simples con texto
        self.icons = {
            "search": "🔍",
            "star": "★",
            "exit": "🚪",
            "info": "ℹ️",
            "product": "📱",
            "beauty": "💄",
            "toy": "🧸",
            "home": "🏠",
            "clothes": "👕"
        }
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Cabecera
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(
            header_frame, 
            text="🌟 Sistema de Búsqueda de Reseñas 🌟", 
            font=("Arial", 18, "bold"),
            foreground=COLOR_SECONDARY
        )
        title_label.pack(pady=10)
        
        # Tarjetas de categorías
        categories_frame = ttk.LabelFrame(
            main_frame, 
            text="Categorías Populares",
            padding=10
        )
        categories_frame.pack(fill=tk.X, padx=5, pady=10)
        
        categories = [
            {"name": "Tecnología", "icon": self.icons["product"], "examples": ["batería larga duración", "pantalla brillante"]},
            {"name": "Belleza", "icon": self.icons["beauty"], "examples": ["crema hidratante", "labial duradero"]},
            {"name": "Juguetes", "icon": self.icons["toy"], "examples": ["educativo", "resistente"]},
            {"name": "Hogar", "icon": self.icons["home"], "examples": ["fácil de limpiar", "ahorro energía"]},
            {"name": "Ropa", "icon": self.icons["clothes"], "examples": ["cómodo", "talla exacta"]}
        ]
        
        for cat in categories:
            cat_frame = ttk.Frame(categories_frame, padding=5)
            cat_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Icono y nombre
            icon_label = ttk.Label(cat_frame, text=cat["icon"], font=("Arial", 14))
            icon_label.pack(anchor=tk.W)
            
            name_label = ttk.Label(cat_frame, text=cat["name"], font=("Arial", 10, "bold"))
            name_label.pack(anchor=tk.W)
            
            # Ejemplos
            for example in cat["examples"]:
                example_label = ttk.Label(
                    cat_frame, 
                    text=f"• {example}", 
                    font=("Arial", 9),
                    foreground=COLOR_TEXT
                )
                example_label.pack(anchor=tk.W, padx=(0, 5))
        
        # Buscador
        search_frame = ttk.Frame(main_frame, padding=10)
        search_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(
            search_frame, 
            textvariable=self.search_var, 
            width=50,
            font=("Arial", 11)
        )
        search_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        search_entry.bind("<Return>", lambda event: self.perform_search())
        
        search_btn = ttk.Button(
            search_frame, 
            text=f"{self.icons['search']} Buscar", 
            command=self.perform_search,
            style="Accent.TButton"
        )
        search_btn.pack(side=tk.LEFT)
        
        # Resultados
        results_frame = ttk.LabelFrame(
            main_frame, 
            text="Resultados",
            padding=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Área de resultados con scroll
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            wrap=tk.WORD,
            font=("Arial", 10),
            bg=COLOR_CARD,
            padx=10,
            pady=10
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.tag_configure("header", foreground=COLOR_PRIMARY, font=("Arial", 12, "bold"))
        self.results_text.tag_configure("subheader", foreground=COLOR_SECONDARY, font=("Arial", 10, "bold"))
        self.results_text.tag_configure("data", foreground=COLOR_TEXT, font=("Arial", 10))
        self.results_text.tag_configure("review", foreground="#34495e", font=("Arial", 10))
        self.results_text.tag_configure("stars", foreground=COLOR_STARS, font=("Arial", 12))
        self.results_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Listo para buscar")
        status_bar = ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(5, 2)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configurar estilos
        self.configure_styles()

        export_btn = ttk.Button(
            self.root,
            text="Exportar Grafo RDF",
            command=self.export_rdf_graph_handler,
            style="Accent.TButton"
        )
        export_btn.pack(pady=(5, 10))
    
    def configure_styles(self):
        style = ttk.Style()
        style.configure("TFrame", background=COLOR_BACKGROUND)
        style.configure("TLabel", background=COLOR_BACKGROUND, foreground=COLOR_TEXT)
        style.configure("TLabelFrame", background=COLOR_BACKGROUND, foreground=COLOR_SECONDARY, font=("Arial", 10, "bold"))
        style.configure("TButton", font=("Arial", 10), background=COLOR_PRIMARY, foreground="white")
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), background=COLOR_ACCENT, foreground="white")
        style.configure("TEntry", fieldbackground="white")
        style.map("Accent.TButton", background=[("active", COLOR_ACCENT), ("pressed", "#c0392b")])
    
    def perform_search(self):
        query = self.search_var.get().strip()
        if not query:
            self.status_var.set("Ingrese una consulta")
            return
        
        self.status_var.set(f"Buscando: '{query}'...")
        self.root.update_idletasks()  # Actualizar la interfaz
        
        try:
            # Limpiar resultados anteriores
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            # Realizar búsqueda
            results = self.query_system.semantic_search(query)
            
            if not results:
                self.results_text.insert(tk.END, "No se encontraron resultados relevantes. Intente con otros términos.", "data")
                self.status_var.set(f"0 resultados para: '{query}'")
            else:
                self.results_text.insert(tk.END, f"{len(results)} resultados encontrados para: '{query}'\n\n", "header")
                self.status_var.set(f"{len(results)} resultados para: '{query}'")
                
                # Mostrar cada resultado
                for i, res in enumerate(results):
                    self.results_text.insert(tk.END, f"\n► Resultado #{i+1} ", "header")
                    self.results_text.insert(tk.END, f"(Relevancia: {res['score']})\n\n", "subheader")
                    
                    # Mostrar datos en formato de tabla
                    for key, value in res['data'].items():
                        if value and value != 'N/A':  # Solo mostrar campos con datos
                            if key == 'Rating':
                                self.results_text.insert(tk.END, f"{key}: ", "subheader")
                                self.results_text.insert(tk.END, f"{value}\n", "stars")
                            else:
                                self.results_text.insert(tk.END, f"{key}: ", "subheader")
                                self.results_text.insert(tk.END, f"{value}\n", "data")
                    
                    # Mostrar texto de la reseña
                    if res['text']:
                        self.results_text.insert(tk.END, "\nReseña:\n", "subheader")
                        # Formatear el texto para que no exceda 80 caracteres por línea
                        wrapped_text = self.wrap_text(res['text'], 80)
                        preview = (wrapped_text[:250] + '...') if len(wrapped_text) > 250 else wrapped_text
                        self.results_text.insert(tk.END, f"{preview}\n", "review")
                    
                    self.results_text.insert(tk.END, "\n" + "="*80 + "\n\n")
        
        except Exception as e:
            self.results_text.insert(tk.END, f"Error en la búsqueda: {str(e)}", "data")
            self.status_var.set(f"Error: {str(e)}")
        
        finally:
            self.results_text.config(state=tk.DISABLED)
    
    def wrap_text(self, text, width):
        """Envuelve el texto en líneas de máximo 'width' caracteres"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 > width:
                lines.append(current_line)
                current_line = word
            else:
                current_line += (" " + word) if current_line else word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)

class UniversalReviewQuerySystem:
    def __init__(self, data_path):
        # Cargar datos y reemplazar NaN
        self.df = pd.read_csv(data_path).fillna('')
        
        if self.df.empty:
            raise ValueError("El dataset está vacío")
        
        # Cargar modelo de idioma en inglés
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Modelo inglés no encontrado. Instale con: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Preprocesar textos para BM25
        self.texts = self.df['text'].astype(str).tolist()
        self.tokenized_texts = self._preprocess_texts()
        
        if self.tokenized_texts:
            self.bm25 = BM25Okapi(self.tokenized_texts)
        else:
            raise ValueError("No hay textos válidos para la búsqueda")
    
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
                     and t != "br"]  # Eliminar <br> HTML
            
            if tokens:
                tokenized.append(tokens)
        
        return tokenized
    
    def semantic_search(self, query, top_n=5):
        """Realiza búsqueda semántica usando BM25"""
        if not hasattr(self, 'bm25') or self.bm25 is None:
            return []
        
        tokens = word_tokenize(query.lower())
        english_stopwords = set(stopwords.words('english'))
        tokens = [t for t in tokens 
                 if t not in english_stopwords 
                 and len(t) > 2]
        
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        min_score = 2.0  # Aumentar umbral de relevancia
        results = []
        for idx in top_indices:
            if scores[idx] > min_score:
                results.append(self._format_result(idx, scores[idx]))
        return results

    def extract_rdf_triples(self):
        """
        Extract RDF-like triples from the reviews.
        Example: (Usuario123, compró, Lavadora X300)
                 (Lavadora X300, tiene_sentimiento, negativo)
        """
        triples = []
        for _, row in self.df.iterrows():
            persona = row.get('ner_persons', '')
            producto = row.get('ner_products', '')
            sentimiento = row.get('sentiment', '')  # assuming you have a sentiment column
            if persona and producto:
                triples.append((persona, 'compró', producto))
            if producto and sentimiento:
                triples.append((producto, 'tiene_sentimiento', sentimiento))
        return triples

    def build_rdf_graph(self):
        """
        Build an in-memory RDF graph from the extracted triples.
        The graph is a dict: (subject, predicate) -> set(objects)
        """
        triples = self.extract_rdf_triples()
        graph = {}
        for subj, pred, obj in triples:
            key = (subj, pred)
            if key not in graph:
                graph[key] = set()
            graph[key].add(obj)
        self.rdf_graph = graph
        return graph

    def query_rdf_graph(self, subject=None, predicate=None, obj=None):
        """
        Query the RDF graph for matches.
        Any of subject, predicate, or object can be None to act as a wildcard.
        Returns a list of matching triples.
        """
        if not hasattr(self, "rdf_graph"):
            self.build_rdf_graph()
        results = []
        for (subj, pred), objects in self.rdf_graph.items():
            if (subject is None or subject == subj) and (predicate is None or predicate == pred):
                for o in objects:
                    if obj is None or obj == o:
                        results.append((subj, pred, o))
        return results

    def semantic_rdf_query(self, query_text):
        """
        Example: "productos con más quejas sobre batería en México"
        Returns list of triples matching likely intent.
        """
        # This is a placeholder: for a real system, use NLP to parse intent
        # For now, simple keyword mapping:
        if "quejas" in query_text or "sentimiento negativo" in query_text:
            # Find products with negative sentiment
            triples = self.query_rdf_graph(predicate="tiene_sentimiento", obj="negativo")
            # Further filter by location if mentioned
            results = []
            for s, p, o in triples:
                # Find if product/location match (assuming you have location info in another triple)
                # For now, just collect products
                results.append((s, p, o))
            return results
        # Fallback: return all triples
        return self.query_rdf_graph()
    
    def advanced_semantic_search(self, product=None, brand=None, sentiment=None, location=None, failure_keyword=None, top_n=10):
        """
        Filter reviews by product, brand, sentiment, location, failure keyword, etc.
        Optionally rank by BM25 if failure_keyword is provided.
        Returns a list of dicts: {review_id, score, data, text, triples}
        """
        df = self.df

        # Filter by each attribute if given
        if product:
            df = df[df['ner_products'].str.contains(product, case=False, na=False)]
        if brand:
            df = df[df['ner_brands'].str.contains(brand, case=False, na=False)]
        if sentiment:
            if 'sentiment' in df.columns:
                df = df[df['sentiment'].str.contains(sentiment, case=False, na=False)]
        if location:
            df = df[df['ner_locations'].str.contains(location, case=False, na=False)]
        
        # BM25 ranking by failure keyword (e.g. "batería", "pantalla", etc.)
        if failure_keyword and not df.empty:
            text_list = df['text'].astype(str).tolist()
            # Tokenize
            tokens = word_tokenize(failure_keyword.lower())
            english_stopwords = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in english_stopwords and len(t) > 2]
            # Use BM25 from those filtered reviews
            tokenized_texts = [word_tokenize(t.lower()) for t in text_list]
            bm25 = BM25Okapi(tokenized_texts)
            scores = bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:top_n]
            selected_rows = df.iloc[top_indices]
        else:
            selected_rows = df.head(top_n)

        # Prepare results
        results = []
        for idx, row in selected_rows.iterrows():
            score = 1.0  # default, or from BM25 if calculated
            triples = []
            persona = row.get('ner_persons', '')
            producto = row.get('ner_products', '')
            sentimiento = row.get('sentiment', '')
            if persona and producto:
                triples.append((persona, 'compró', producto))
            if producto and sentimiento:
                triples.append((producto, 'tiene_sentimiento', sentimiento))
            # Use previous _format_result for other data
            data = self._format_result(idx, score)
            data['triples'] = triples
            results.append(data)
        return results

    def visualize_product_sentiment_graph(self):
        """
        Visualize the graph of products and their associated sentiments.
        """
        if not hasattr(self, "rdf_graph"):
            self.build_rdf_graph()
        G = nx.DiGraph()
        # Only add product-sentiment edges
        for (subj, pred), objs in self.rdf_graph.items():
            if pred == "tiene_sentimiento":
                for obj in objs:
                    G.add_edge(subj, obj)
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, k=0.7)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', font_size=10, font_family='Arial')
        plt.title("Relaciones Producto-Sentimiento")
        plt.show()

    def export_rdf_graph_handler(self):
        try:
            # Ask user for a filename/location
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Guardar Grafo RDF como..."
            )
            if not filename:
                return  # User cancelled
            export_path = self.query_system.export_rdf_triples_csv(filename)
            messagebox.showinfo("Exportación exitosa", f"Grafo RDF exportado como:\n{export_path}")
        except Exception as e:
            messagebox.showerror("Error de exportación", f"No se pudo exportar el grafo RDF:\n{str(e)}")
    
    def _format_result(self, idx, score):
        """Formatea un resultado individual con la nueva estructura"""
        row = self.df.iloc[idx]
        
        # Función para obtener valores limpios
        get_val = lambda col: row[col] if col in row and pd.notna(row[col]) and row[col] != '' else 'N/A'
        
        # Obtener valores
        producto = get_val('ner_products')
        marca = get_val('ner_brands')
        lugar = get_val('ner_locations')
        persona = get_val('ner_persons')
        precio = get_val('extracted_prices')
        fecha = get_val('extracted_purchase_dates')
        modelo = get_val('extracted_product_models')
        rating = get_val('rating')  # Columna de valoración
        
        # Construir evento
        evento = ""
        if persona != 'N/A' and producto != 'N/A' and fecha != 'N/A':
            evento = f"{persona} compró {producto} en {fecha}"
        elif persona != 'N/A' and producto != 'N/A':
            evento = f"{persona} compró {producto}"
        elif producto != 'N/A' and fecha != 'N/A':
            evento = f"Compró {producto} en {fecha}"
        elif producto != 'N/A':
            evento = f"Compra de {producto}"
        else:
            evento = "Información de compra no disponible"
        
        # Convertir rating a estrellas si es numérico
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
        
        # Crear diccionario con la estructura deseada
        table_data = {
            'Producto': producto,
            'Marca': marca,
            'Lugar': lugar,
            'Persona': persona,
            'Precio': precio,
            'Fecha': fecha,
            'Modelo': modelo,
            'Rating': rating_stars,  # Valoración en formato de estrellas
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
    data_path = './Data/processed_data/Dataset_processed_ner.csv'
    if not os.path.exists(data_path):
        messagebox.showerror("Error", f"Archivo no encontrado: {data_path}\nVerifique la ruta o ejecute primero el procesamiento de datos")
    else:
        root = tk.Tk()
        app = ReviewSearchApp(root, data_path)
        
        # Centrar ventana en la pantalla
        window_width = 1100
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        root.mainloop()