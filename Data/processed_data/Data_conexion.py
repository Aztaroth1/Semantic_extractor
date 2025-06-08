import pandas as pd

# Rutas a tus archivos CSV
csv1 = 'Data\processed_data\Electronics_processed_ner.csv'
csv2 = 'Data\processed_data\Beauty_processed_ner.csv'
csv3 = 'Data\processed_data\Music_Instruments_sm_processed.csv'

# Leer los tres CSV
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
df3 = pd.read_csv(csv3)

# Unirlos en un solo DataFrame
df_total = pd.concat([df1, df2, df3], ignore_index=True)

# (Opcional) Verifica que se unieron correctamente
print(f"Total de filas combinadas: {len(df_total)}")
print(df_total.head())

# Guardar el CSV combinado
df_total.to_csv('Dataset_processed_ner.csv', index=False)
print("CSV combinado guardado")
