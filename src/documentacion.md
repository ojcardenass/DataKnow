# Documentación

## Test 1:

- **Definición de la Carpeta Base:**
  - Se establece la carpeta base para la apertura o guardado de archivos.

- **Manejo del Archivo TXT:**
  - Abre el archivo utilizando el bloque 'with' para gestionar automáticamente el archivo.
  - Define una función `process_txt` para procesar el archivo TXT, reemplazando los ':' por ',', eliminando espacios en blanco, agregando el agente a todas las plantas correspondientes, eliminando elementos vacíos y separando los elementos por ',', convirtiendo así el archivo en una lista de listas.
  - Define el nombre de las columnas.
  - Crea un DataFrame con los nombres de las columnas y los datos procesados.
  - Filtra el DataFrame por las plantas tipo D.
  - Elimina la columna 'Tipo'.
  - Convierte las columnas 'Hora_i' a numéricas.
  - Guarda el DataFrame como un archivo XLSX.

## Test 2: Manipulación de Datos

- **Definición de la Ruta Base y Archivo de Datos Maestros:**
  - Define la ruta base junto con el archivo de datos maestros.

- **Lectura del Archivo de Datos Maestros:**
  - Lee el archivo de datos maestros con el método `read_excel` de pandas, utilizando los parámetros 'sheet_name' y 'usecols' para seleccionar los datos de la hoja 'Master Data Oficial' y las columnas 'AGENTE (OFEI)', 'CENTRAL (dDEC, dSEGDES, dPRU…)', 'Tipo de central (Hidro, Termo, Filo, Menor)' respectivamente.

- **Manipulación de Datos:**
  - Renombra las columnas para facilitar la manipulación.
  - Filtra la data correspondiente a agente = EMGESA ó EMGESA S.A. y Tipo de Central ‘H’ o ‘T’.
  - Carga la data del archivo dDEC1204.TXT que tiene los datos en el formato Central, Hora_1, Hora_2, …, Hora_24.
  - Une los DataFrames basándose en la columna 'Central'.
  - Agrupa la data por Agente, Central, Tipo y suma los valores de las columnas 'Hora_i'.
  - Suma las filas en la columna 'Total'.
  - Filtra la data con 'Total' > 0.
  - Guarda el DataFrame como un archivo XLSX.

## Test 3: SQL

- **Definición de Función SQL:**
  - Define una función que recibe un query y un cursor como parámetros, retornando un DataFrame de pandas con el resultado de la consulta.

- **Creación de Base de Datos en Memoria:**
  - Crea una base de datos en memoria.

- **Ejecución de Scripts SQL:**
  - Crea un objeto cursor.
  - Ejecuta un script con el cursor para crear la tabla y realizar las inserciones.

- **Ejecución de Consultas SQL:**
  - Ejecuta consultas y guarda los resultados en DataFrames.

- **Exportación a Excel:**
  - Exporta los DataFrames a un archivo Excel con diferentes hojas.