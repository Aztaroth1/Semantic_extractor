from datasets import load_dataset
import pandas as pd
import itertools
# Cargar la referencia al dataset (esto es rápido)
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", trust_remote_code=True)
# Seleccionar solo las primeras 100,000 filas
full_list = list(itertools.islice(dataset['full'], 100000))

porcion_dataset = pd.DataFrame(full_list)

porcion_dataset.to_csv("Electronics.csv", index=False)
# Es buena idea cambiarle el nombre para saber que es una porción

print("¡Se guardó una porción de 10,000 reseñas en 'Music_Instruments_primeras_10000.csv'!")