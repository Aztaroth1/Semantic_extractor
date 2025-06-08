import pandas as pd
import re
import numpy as np
from datetime import datetime
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed

class BeautyReviewsCleaner:
    def __init__(self, file_path, max_workers=6):
        self.file_path = file_path
        self.df = None
        self.max_workers = max_workers

        try:
            self.nlp = spacy.load('en_core_web_sm', disable=["parser", "tagger", "lemmatizer"])
            print("Modelo spaCy cargado.")
        except OSError:
            print("Error: Ejecuta 'python -m spacy download en_core_web_sm'")
            self.nlp = None

        self.price_patterns = [re.compile(p, re.IGNORECASE) for p in [
            r'\$\s*(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*dollars?',
            r'(\d+(?:\.\d{2})?)\s*\$',
            r'price\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',
            r'cost\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',
            r'paid\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)'
        ]]

        self.date_patterns = [re.compile(p, re.IGNORECASE) for p in [
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{2,4}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{1,2},?\s+\d{2,4}',
            r'(?:purchased|bought|ordered)\s+(?:on\s+)?([a-zA-Z]+\s+\d{1,2},?\s+\d{2,4})'
        ]]

        self.model_patterns = [re.compile(p, re.IGNORECASE) for p in [
            r'\bmodel\s*#?:?\s*([A-Z0-9\-]+)',
            r'\bproduct\s*(?:id|code|number)\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'\bSKU\s*[:\-]?\s*([A-Z0-9\-]+)',
            r'\b[A-Z]{2,}-\d{2,}[A-Z]?\b',
            r'\b\d{3,}[A-Z]{1,}\b',
            r'\b[A-Z]{2,}\d{2,}\b',
            r'\b[A-Z]+\d+[A-Z]*\b'
        ]]

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset cargado: {self.df.shape[0]} filas.")
            return True
        except Exception as e:
            print(f"Error al cargar CSV: {e}")
            return False

    def clean_text(self, text):
        if pd.isna(text): return ''
        text = str(text)

        patterns = [
            r'http\S+', r'\b[\w.-]+@[\w.-]+\.\w+\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'<[^>]+>', r'[^\w\s\.\,\!\?\-\'\/\$\:\(\)]'
        ]
        emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticonos
        u"\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        u"\U0001F680-\U0001F6FF"  # transporte y mapas
        u"\U0001F1E0-\U0001F1FF"  # banderas
        u"\U00002700-\U000027BF"  # otros símbolos
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
        for p in patterns:
            text = re.sub(p, ' ', text)

        for p in patterns:
            text = re.sub(p, ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_prices(self, text):
        if pd.isna(text): return ''
        text = text.lower()
        prices = [float(m) for pattern in self.price_patterns for m in pattern.findall(text) if m]
        return ', '.join(f"${p:.2f}" for p in sorted(set(prices))) if prices else ''

    def extract_dates(self, text):
        if pd.isna(text): return ''
        text = text.lower()
        matches = [m.strip() for p in self.date_patterns for m in p.findall(text)]
        return ', '.join(sorted(set(matches))) if matches else ''

    def extract_models(self, text):
        if pd.isna(text): return ''
        matches = []
        for p in self.model_patterns:
            found = p.findall(text)
            matches.extend(found)
        filtered = [m.strip() for m in matches if m and not re.fullmatch(r'\d+', m)]
        return ', '.join(list(set(filtered))[:5]) if filtered else ''

    def extract_entities_batch(self, texts):
        if self.nlp is None:
            return [{'Producto': '', 'Marca': '', 'Lugar': '', 'Persona': ''} for _ in texts]

        label_map = {'PRODUCT': 'Producto', 'ORG': 'Marca', 'GPE': 'Lugar', 'PERSON': 'Persona'}
        results = []

        for doc in self.nlp.pipe(texts, batch_size=1000, n_process=1):
            result = {v: set() for v in label_map.values()}
            for ent in doc.ents:
                label = label_map.get(ent.label_)
                if label:
                    result[label].add(ent.text.strip())
            results.append({k: ', '.join(v) for k, v in result.items()})
        return results

    def process_reviews(self):
        if self.df is None or self.nlp is None:
            print("Error: Dataset o modelo spaCy no disponible.")
            return

        df = self.df.copy()
        print("Limpiando texto...")
        df['text'] = df['text'].map(self.clean_text)
        df = df[df['text'].str.len() > 0]

        print("Extrayendo precios, fechas y modelos en paralelo...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                'prices': executor.map(self.extract_prices, df['text']),
                'dates': executor.map(self.extract_dates, df['text']),
                'models': executor.map(self.extract_models, df['text'])
            }

        df['extracted_prices'] = list(futures['prices'])
        df['extracted_purchase_dates'] = list(futures['dates'])
        df['extracted_product_models'] = list(futures['models'])

        print("Extrayendo entidades con spaCy.pipe...")
        ner_results = self.extract_entities_batch(df['text'].tolist())
        df['ner_products'] = [r['Producto'] for r in ner_results]
        df['ner_brands'] = [r['Marca'] for r in ner_results]
        df['ner_locations'] = [r['Lugar'] for r in ner_results]
        df['ner_persons'] = [r['Persona'] for r in ner_results]

        self.df = df
        print(f"Procesamiento completado: {len(df)} opiniones procesadas.")

    def save_processed_data(self, output_file):
        if self.df is not None:
            try:
                self.df.to_csv(output_file, index=False)
                print(f"Archivo guardado: {output_file}")
            except Exception as e:
                print(f"Error al guardar archivo: {e}")
        else:
            print("No hay datos para guardar.")

    def get_processing_summary(self):
        if self.df is None:
            print("No hay datos procesados.")
            return {}

        resumen = {
            'Total de opiniones': len(self.df),
            'Con precios': self.df['extracted_prices'].astype(bool).sum(),
            'Con fechas': self.df['extracted_purchase_dates'].astype(bool).sum(),
            'Con modelos': self.df['extracted_product_models'].astype(bool).sum(),
            'Con productos (NER)': self.df['ner_products'].astype(bool).sum(),
            'Con marcas (NER)': self.df['ner_brands'].astype(bool).sum(),
            'Con lugares (NER)': self.df['ner_locations'].astype(bool).sum(),
            'Con personas (NER)': self.df['ner_persons'].astype(bool).sum(),
        }

        print("\nResumen del procesamiento:")
        for k, v in resumen.items():
            print(f"  - {k}: {v}")
        return resumen
if __name__ == "__main__":
    cleaner = BeautyReviewsCleaner("Data\\Raw_data\\Beauty_40k.csv")
    if cleaner.load_data():
        cleaner.process_reviews()
        cleaner.save_processed_data("Data\\processed_data\\Beauty_40k_processed_ner.csv")
        cleaner.get_processing_summary()
