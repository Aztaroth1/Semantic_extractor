import pandas as pd
from datasets import load_dataset
import re
import numpy as np
from datetime import datetime
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Para que el resultado sea consistente


class BeautyReviewsCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset cargado exitosamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
            return True
        except Exception as e:
            print(f"Error al cargar el dataset: {e}")
            return False

    def is_english(self, text):
        try:
            return detect(text) == 'en'
        except:
            return False

    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text)
        text = re.sub(r"[^a-zA-Z0-9\s\$\€\£\.\,\-/\:\(\)%]", " ", text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\bVIDEOID:[a-fA-F0-9]{32}\b', '', text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\/\$\:\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def extract_prices(self, text):
        if pd.isna(text) or text == '':
            return ''
        price_patterns = [
            r'\$\s*(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*dollars?',
            r'(\d+(?:\.\d{2})?)\s*\$',
            r'price\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',
            r'cost\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',
            r'paid\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',
        ]
        prices = []
        text_lower = text.lower()
        for pattern in price_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            prices.extend([match for match in matches if match])
        unique_prices = list(set([float(p) for p in prices if p]))
        unique_prices.sort()
        return ', '.join([f'${p:.2f}' for p in unique_prices])

    def extract_purchase_dates(self, text):
        if pd.isna(text) or text == '':
            return ''
        date_patterns = [
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{2,4}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{2,4}',
            r'purchased\s+(?:on\s+)?([a-zA-Z]+\s+\d{1,2},?\s+\d{2,4})',
            r'bought\s+(?:on\s+)?([a-zA-Z]+\s+\d{1,2},?\s+\d{2,4})',
            r'ordered\s+(?:on\s+)?([a-zA-Z]+\s+\d{1,2},?\s+\d{2,4})',
        ]
        dates = []
        text_lower = text.lower()
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            dates.extend([match.strip() for match in matches if match.strip()])
        unique_dates = list(set(dates))
        return ', '.join(unique_dates)

    def extract_product_models(self, text):
        if pd.isna(text) or text == '':
            return ''
        model_patterns = [
            "fit me", "superstay", "instant age rewind", "true match",
            "infallible", "studio fix", "colorstay", "soft matte",
            "pro filt’r", "niacinamide", "aha 30%", "bha 2%",
            "foaming cleanser", "hydro boost", "effaclar duo",
            "fructis", "keratin smooth", "oil repair", "hair food",
            "spf50", "spf30", "bb cream", "cc cream",
            "nc20", "nc30", "nw40", "120 classic ivory",
            "230", "shade 10", "tono claro",
            r'\b(?:MAC|Urban Decay|Maybelline|L\'Oreal|Revlon|Clinique|Estee Lauder|Dior|Chanel|YSL)\s+([A-Z][a-zA-Z0-9\s\-\']+\d*)',
            r'\b(?:Fender|Gibson|Yamaha|Ibanez|Epiphone|Martin|Taylor)\s+([A-Z][a-zA-Z0-9\s\-]+)',
            r'\b(?:iPhone)\s+(\d{1,2}(?:\s+Pro)?(?:\s+Max)?)',
        r'\b(?:Samsung\s+Galaxy)\s+([A-Z]\d{1,2}(?:\+|\s+Ultra)?)',
        r'\b(?:Google\s+Pixel)\s+(\d{1,2}(?:\s+Pro)?)',
        # Laptops
        r'\b(?:MacBook)\s+(Air|Pro)\s*(\d{2})?',
        r'\b(?:ThinkPad|Inspiron|Pavilion)\s+([A-Z0-9][a-zA-Z0-9\s\-]*)',
        # TVs
        r'\b(?:Samsung|LG|Sony)\s+(\d{2,3}[\"\']\s*[A-Z0-9]+)'
        ]
        models = []
        for pattern in model_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Si es tupla (varios grupos), toma los no vacíos
                if isinstance(match, tuple):
                    for m in match:
                        if m and isinstance(m, str) and m.strip() and len(m.strip()) > 1:
                            models.append(m.strip())
                else:
                    if match and isinstance(match, str) and match.strip() and len(match.strip()) > 1:
                        models.append(match.strip())
        clean_models = []
        for model in models:
            if not re.match(r'^\d+$', model):
                clean_models.append(model)
        unique_models = list(set(clean_models))
        return ', '.join(unique_models[:5])

    def process_reviews(self):
        if self.df is None:
            print("Error: Debe cargar el dataset primero")
            return
        print("Iniciando procesamiento de reviews...")
        processed_df = self.df.copy()

        print("Filtrando reviews no escritas en inglés...")
        english_mask = processed_df['text'].apply(self.is_english)
        initial_lang_count = len(processed_df)
        processed_df = processed_df[english_mask]
        lang_filtered_count = len(processed_df)
        print(f"Reviews eliminadas por no estar en inglés: {initial_lang_count - lang_filtered_count}")

        # Eliminar textos vacíos
        processed_df = processed_df[processed_df['text'].str.len() > 0]

        # Eliminar textos con menos de 150 palabras
        print("Eliminando reviews con menos de 30 palabras...")
        processed_df = processed_df[processed_df['text'].apply(lambda x: len(x.split())) >= 30]
        print("Extrayendo precios...")
        processed_df['extracted_prices'] = processed_df['text'].apply(self.extract_prices)

        print("Extrayendo fechas de compra...")
        processed_df['extracted_purchase_dates'] = processed_df['text'].apply(self.extract_purchase_dates)

        print("Extrayendo modelos de productos...")
        processed_df['extracted_product_models'] = processed_df['text'].apply(self.extract_product_models)

        print("Limpiando texto (manteniendo números y símbolos para precios/fechas)...")
        processed_df['text'] = processed_df['text'].apply(self.clean_text)

        initial_count = len(processed_df)

        

        final_count = len(processed_df)

        print("Procesamiento completado.")
        print(f"Reviews removidas por quedar vacías o tener menos de 50 palabras: {initial_count - final_count}")
        print(f"Reviews finales: {final_count}")

        self.df = processed_df

    def save_processed_data(self, output_path='Data/processed_data/Electronics_processed.csv'):
        if self.df is None:
            print("Error: No hay datos para guardar")
            return
        self.df.to_csv(output_path, index=False)
        print(f"Dataset procesado guardado en: {output_path}")
        print(f"\nColumnas en el archivo procesado:")
        for col in self.df.columns:
            print(f"  - {col}")
        print(f"\nPrimeras 3 filas de ejemplo:")
        print(self.df[['text', 'extracted_prices', 'extracted_purchase_dates', 'extracted_product_models']].head(3))

    def get_processing_summary(self):
        if self.df is None:
            return
        total_reviews = len(self.df)
        reviews_with_prices = len(self.df[self.df['extracted_prices'] != ''])
        reviews_with_dates = len(self.df[self.df['extracted_purchase_dates'] != ''])
        reviews_with_models = len(self.df[self.df['extracted_product_models'] != ''])
        print("\n" + "=" * 50)
        print("RESUMEN DEL PROCESAMIENTO")
        print("=" * 50)
        print(f"Total de reviews procesadas: {total_reviews}")
        print(f"Reviews con precios extraídos: {reviews_with_prices} ({reviews_with_prices / total_reviews * 100:.1f}%)")
        print(f"Reviews con fechas extraídas: {reviews_with_dates} ({reviews_with_dates / total_reviews * 100:.1f}%)")
        print(f"Reviews con modelos extraídos: {reviews_with_models} ({reviews_with_models / total_reviews * 100:.1f}%)")


def main():
    cleaner = BeautyReviewsCleaner('Electronics.csv')
    if not cleaner.load_data():
        return
    cleaner.process_reviews()
    cleaner.get_processing_summary()
    cleaner.save_processed_data()
    print("\n¡Proceso completado exitosamente!")
    print("El archivo 'Beauty_40k_processed.csv' contiene:")
    print("- Columna 'text': texto completamente limpio")
    print("- Columna 'extracted_prices': precios encontrados")
    print("- Columna 'extracted_purchase_dates': fechas de compra encontradas")
    print("- Columna 'extracted_product_models': modelos/especificaciones encontrados")


if __name__ == "__main__":
    main()
