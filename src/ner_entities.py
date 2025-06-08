import pandas as pd
import re
import spacy
import time

class BeautyReviewsCleaner:
    def __init__(self):
        print("Inicializando el procesador (solo modelo EN)...")
        self.nlp_en = None

        try:
            self.nlp_en = spacy.load('en_core_web_sm', disable=['parser'])
            print("Modelo en INGLÉS ('en_core_web_sm') cargado.")
        except OSError as e:
            print(f"Error al cargar el modelo: {e}")

    def process_reviews(self, df):
        if not self.nlp_en:
            print("Error: El modelo de spaCy no está cargado.")
            return None

        if 'text' not in df.columns:
            print("Error: El DataFrame debe contener una columna llamada 'text'.")
            return None
        
        if 'title' not in df.columns:
            print("Advertencia: La columna 'title' no fue encontrada. Se usará solo 'text'.")
            df['title'] = ''

        # Combinar title y text para análisis
        df['full_text'] = (df['title'].fillna('') + '. ' + df['text'].fillna('')).str.strip()

        texts = df['full_text'].tolist()
        results_list = []


        print(f"Iniciando procesamiento de {len(texts)} textos...")
        start_time = time.time()

        docs_en = list(self.nlp_en.pipe(texts, n_process=-1, batch_size=500))

        for i, doc_en in enumerate(docs_en):
            original_text = texts[i]
            products = set()
            brands = set()
            locations = set()
            persons = set()

            for ent in doc_en.ents:
                if ent.label_ == 'PRODUCT': products.add(ent.text.strip())
                elif ent.label_ == 'ORG': brands.add(ent.text.strip())
                elif ent.label_ == 'GPE': locations.add(ent.text.strip())
                elif ent.label_ == 'PERSON': persons.add(ent.text.strip())

           
            results_list.append({
                'text': df.loc[i, 'text'],
                'title': df.loc[i, 'title'],
                'ner_products': ', '.join(sorted(products)),
                'ner_brands': ', '.join(sorted(brands)),
                'ner_locations': ', '.join(sorted(locations)),
                'ner_persons': ', '.join(sorted(persons)),
            })

        end_time = time.time()
        print(f"Procesamiento completado en {end_time - start_time:.2f} segundos.")

        processed_df = pd.DataFrame(results_list)
        original_cols_df = df.drop(columns=['full_text'], errors='ignore')
        final_df = pd.concat([original_cols_df, processed_df.drop(columns=['text', 'title'], errors='ignore')], axis=1)
        return final_df

def main():
    file_path = 'Data/processed_data/Electronics_processed.csv'
    
    print(f"Cargando dataset desde: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset cargado exitosamente. {len(df)} filas encontradas.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta '{file_path}'.")
        return
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo CSV: {e}")
        return

    cleaner = BeautyReviewsCleaner()
    
    if cleaner.nlp_en:
        processed_df = cleaner.process_reviews(df)
        if processed_df is not None:
            print("\n--- Procesamiento finalizado. Mostrando 5 filas de ejemplo: ---")
            display_cols = [
                'title', 'text', 'ner_brands', 'ner_locations', 'extracted_prices', 'extracted_product_models'
            ]
            existing_display_cols = [col for col in display_cols if col in processed_df.columns]
            print(processed_df[existing_display_cols].head())

            output_path = 'Data/Processed_data/Electronics_processed_ner.csv'
            processed_df.to_csv(output_path, index=False)
            print(f"\nDataset procesado y guardado exitosamente en: {output_path}")

if __name__ == "__main__":
    main()
