import pandas as pd
from pathlib import Path

def create_data_dictionary(df, confidential_columns=None):
    """
    Genera un diccionario de datos completo a partir de un DataFrame.
    """
    if confidential_columns is None:
        confidential_columns = []

    data = []
    total_rows = len(df)

    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0

        data.append({
            'Column Name': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Unique Values': df[col].nunique(),
            'Missing Values': missing_count,
            '% Missing': f'{missing_percentage:.2f}%',
            'Confidential': 'Sí' if col in confidential_columns else 'No'
        })

    return pd.DataFrame(data)


# 🔥 FUNCIÓN AUTOMÁTICA PARA TODO EL PROYECTO
def generate_project_data_dictionary(data_path="splits", output_path="data_dictionary"):
    """
    Genera diccionarios de datos para train, val y test automáticamente
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    files = ["train.csv", "val.csv", "test.csv"]

    # ⚠️ En tu proyecto estas son las columnas sensibles (puedes ajustar)
    confidential_cols = []  # normalmente no hay datos sensibles aquí

    for file in files:
        file_path = data_path / file

        if not file_path.exists():
            print(f"⚠️ Archivo no encontrado: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Generar diccionario
        data_dict = create_data_dictionary(df, confidential_columns=confidential_cols)

        # Guardar resultado
        output_file = output_path / f"data_dictionary_{file.replace('.csv','')}.csv"
        data_dict.to_csv(output_file, index=False)

        print(f"✅ Diccionario generado: {output_file}")


# 🚀 EJECUCIÓN
if __name__ == "__main__":
    generate_project_data_dictionary()
    