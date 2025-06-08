import pandas as pd
import re
import numpy as np
from datetime import datetime

class BeautyReviewsCleaner:
    def __init__(self, file_path):
        """
        Inicializa el limpiador de reviews
        
        Args:
            file_path (str): Ruta al archivo CSV
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Carga el dataset CSV"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset cargado exitosamente: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
            return True
        except Exception as e:
            print(f"Error al cargar el dataset: {e}")
            return False
    
    def clean_text(self, text):
        """
        Realiza limpieza básica del texto manteniendo números y caracteres para precios y fechas
        
        Args:
            text (str): Texto a limpiar
            
        Returns:
            str: Texto limpio manteniendo números y símbolos relevantes
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convertir a string
        text = str(text)
        
        # Remover URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remover emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remover números de teléfono (formato específico para no afectar fechas/precios)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remover HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Mantener: letras, números, espacios, puntuación básica, símbolos de precio/fecha
        # Remover solo caracteres especiales problemáticos
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\/\$\:\(\)]', ' ', text)
        
        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Remover espacios al inicio y final
        text = text.strip()
        
        return text
    
    def extract_prices(self, text):
        """
        Extrae precios mencionados en el texto original
        
        Args:
            text (str): Texto original donde buscar precios
            
        Returns:
            str: Precios encontrados separados por coma
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Patrones para precios
        price_patterns = [
            r'\$\s*(\d+(?:\.\d{2})?)',  # $19.99, $19
            r'(\d+(?:\.\d{2})?)\s*dollars?',  # 19.99 dollars
            r'(\d+(?:\.\d{2})?)\s*\$',  # 19.99$
            r'price\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',  # price: $19.99
            r'cost\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',  # cost: $19.99
            r'paid\s*:?\s*\$?\s*(\d+(?:\.\d{2})?)',  # paid: $19.99
        ]
        
        prices = []
        text_lower = text.lower()
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            prices.extend([match for match in matches if match])
        
        # Remover duplicados y convertir a float para ordenar
        unique_prices = list(set([float(p) for p in prices if p]))
        unique_prices.sort()
        
        return ', '.join([f'${p:.2f}' for p in unique_prices])
    
    def extract_purchase_dates(self, text):
        """
        Extrae fechas de compra mencionadas en el texto original
        
        Args:
            text (str): Texto original donde buscar fechas
            
        Returns:
            str: Fechas encontradas separadas por coma
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Patrones para fechas
        date_patterns = [
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',  # MM/DD/YYYY, MM-DD-YY
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
        
        # Remover duplicados
        unique_dates = list(set(dates))
        
        return ', '.join(unique_dates)
    
    def extract_product_models(self, text):
        """
        Extrae modelos, colores, tamaños y especificaciones de productos
        
        Args:
            text (str): Texto original donde buscar modelos
            
        Returns:
            str: Modelos encontrados separados por coma
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Patrones para modelos de productos
        model_patterns = [
    r'\b(?:model|modelo)\s*:?\s+((?=.*[\d\-/._])[A-Z0-9][A-Z0-9\s\-./_]+[A-Z0-9])',
    r'\b(?:ref|reference|referencia)\s*[:.]?\s+((?=.*[\d\-/._])[A-Z0-9][A-Z0-9\-_/]+)',
    r'\b(?:SKU|P/N|Part No|Art|Item(?:\s*#|\s*No)?)[\s:.]+\s*((?=.*[\d\-/._])[A-Z0-9][A-Z0-9\-/]+)',
    r'\b(?:version|versión)\s*:?\s+([A-Z0-9\s.\-]+[A-Z0-9])',
    r'\b(?:colou?r|shade|tono)\s*:?\s+([A-Z0-9\-]+(?:\s+[A-Z0-9\-]+)+)',
    r'\b([A-Z]{2,4}[-.]\s?[A-Z0-9]{3,})',
    r'\b([A-Z0-9]+(?:[-/][A-Z0-9]+)+)',
    r'\b([A-Z]{2,5}\d{2,6}[A-Z]?)\b',
    r'\b(\d{4,8}[A-Z]{1,2})\b',
    r'(\b\d+(?:[.,]\d+)?\s*(?:oz|ml|g|l|kg|fl\s*oz|mm|cm|m|inch(?:es)?)\b)',
]
        
        models = []
        
        for pattern in model_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            models.extend([match.strip() for match in matches if match.strip() and len(match.strip()) > 1])
        
        # Limpiar y filtrar modelos
        clean_models = []
        for model in models:
            # Remover números solos
            if not re.match(r'^\d+$', model):
                clean_models.append(model)
        
        # Remover duplicados
        unique_models = list(set(clean_models))
        
        return ', '.join(unique_models[:5])  # Limitar a 5 modelos máximo
    
    def process_reviews(self):
        """
        Procesa todas las reviews aplicando limpieza y extracción
        """
        if self.df is None:
            print("Error: Debe cargar el dataset primero")
            return
        
        print("Iniciando procesamiento de reviews...")
        
        # Crear copia del dataframe original
        processed_df = self.df.copy()
        
        # Extraer información ANTES de limpiar el texto
        print("Extrayendo precios...")
        processed_df['extracted_prices'] = processed_df['text'].apply(self.extract_prices)
        
        print("Extrayendo fechas de compra...")
        processed_df['extracted_purchase_dates'] = processed_df['text'].apply(self.extract_purchase_dates)
        
        print("Extrayendo modelos de productos...")
        processed_df['extracted_product_models'] = processed_df['text'].apply(self.extract_product_models)
        
        # Ahora limpiar el texto manteniendo números y símbolos importantes
        print("Limpiando texto (manteniendo números y símbolos para precios/fechas)...")
        processed_df['text'] = processed_df['text'].apply(self.clean_text)
        
        # Remover reviews que quedaron vacías después de la limpieza
        initial_count = len(processed_df)
        processed_df = processed_df[processed_df['text'].str.len() > 0]
        final_count = len(processed_df)
        
        print(f"Procesamiento completado.")
        print(f"Reviews removidas por quedar vacías: {initial_count - final_count}")
        print(f"Reviews finales: {final_count}")
        
        self.df = processed_df
        
    def save_processed_data(self, output_path='Data\processed_data\Beauty_40k_processed.csv'):
        """
        Guarda el dataset procesado
        
        Args:
            output_path (str): Ruta de salida del archivo procesado
        """
        if self.df is None:
            print("Error: No hay datos para guardar")
            return
        
        self.df.to_csv(output_path, index=False)
        print(f"Dataset procesado guardado en: {output_path}")
        
        # Mostrar información del archivo generado
        print(f"\nColumnas en el archivo procesado:")
        for col in self.df.columns:
            print(f"  - {col}")
        
        print(f"\nPrimeras 3 filas de ejemplo:")
        print(self.df[['text', 'extracted_prices', 'extracted_purchase_dates', 'extracted_product_models']].head(3))
    
    def get_processing_summary(self):
        """
        Obtiene resumen del procesamiento
        """
        if self.df is None:
            return
        
        total_reviews = len(self.df)
        reviews_with_prices = len(self.df[self.df['extracted_prices'] != ''])
        reviews_with_dates = len(self.df[self.df['extracted_purchase_dates'] != ''])
        reviews_with_models = len(self.df[self.df['extracted_product_models'] != ''])
        
        print("\n" + "="*50)
        print("RESUMEN DEL PROCESAMIENTO")
        print("="*50)
        print(f"Total de reviews procesadas: {total_reviews}")
        print(f"Reviews con precios extraídos: {reviews_with_prices} ({reviews_with_prices/total_reviews*100:.1f}%)")
        print(f"Reviews con fechas extraídas: {reviews_with_dates} ({reviews_with_dates/total_reviews*100:.1f}%)")
        print(f"Reviews con modelos extraídos: {reviews_with_models} ({reviews_with_models/total_reviews*100:.1f}%)")

def main():
    """Función principal para ejecutar el procesamiento"""
    
    # Inicializar el procesador
    cleaner = BeautyReviewsCleaner('Data\Raw_data\Beauty_40k.csv')
    
    # Cargar datos
    if not cleaner.load_data():
        return
    
    # Procesar reviews
    cleaner.process_reviews()
    
    # Mostrar resumen
    cleaner.get_processing_summary()
    
    # Guardar datos procesados
    cleaner.save_processed_data()
    
    print("\n¡Proceso completado exitosamente!")
    print("El archivo 'Beauty_40k_processed.csv' contiene:")
    print("- Columna 'text': texto completamente limpio")
    print("- Columna 'extracted_prices': precios encontrados")
    print("- Columna 'extracted_purchase_dates': fechas de compra encontradas")
    print("- Columna 'extracted_product_models': modelos/especificaciones encontrados")
    print("- Todas las demás columnas originales")

if __name__ == "__main__":
    main()