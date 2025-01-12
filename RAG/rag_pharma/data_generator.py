import pandas as pd
import numpy as np

# Definir los productos y precios
productos = {
    "Ibuprofeno": 5.00,
    "Lorazepam": 10.00,
    "Minoxidil": 15.00,
    "Almax": 3.00
}

# Crear una lista para almacenar los datos
data = []

# Generar datos para cada día de diciembre
for dia in range(1, 32):
    fecha = f"2023-12-{dia:02d}"
    for producto, precio in productos.items():
        cantidad_vendida = np.random.randint(1, 20)  # Simular ventas entre 1 y 20
        total_ventas = cantidad_vendida * precio
        data.append([fecha, producto, cantidad_vendida, precio, total_ventas])

# Crear un DataFrame
df = pd.DataFrame(data, columns=["Fecha", "Producto", "Cantidad Vendida", "Precio Unitario", "Total Ventas"])

# Guardar el DataFrame en un archivo Excel
df.to_excel("ventas_farmacia_diciembre.xlsx", index=False)