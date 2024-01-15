import sqlite3
import pandas as pd

# A function that takes a query and returns a pandas data frame
def execute_query(query, cursor) -> pd.DataFrame:
    # Use the cursor to execute the query
    cursor.execute(query)
    # Fetch the data
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    # Create a pandas dataframe
    df = pd.DataFrame(rows, columns=columns)
    return df

# Create a database in RAM
conn = sqlite3.connect(':memory:')
# Create a cursor object
cursor = conn.cursor()

# Execute the CREATE TABLE and INSERT statements
cursor.executescript('''
    CREATE TABLE EMPLEADO (
        ID INT(8),
        NOMBRE VARCHAR(50),
        APELLIDO VARCHAR(59),
        SEXO CHAR(1),
        FECHA_NACIMIENTO DATE,
        SALARIO DOUBLE(10,2)
    );

    CREATE TABLE VACACIONES(
        ID INT(8),
        ID_EMP INT(8),
        FECHA_INICIO DATE,
        FECHA_FIN DATE,
        ESTADO CHAR(1),
        CANTIDAD_DIAS INT(8)
    );

    INSERT INTO EMPLEADO VALUES (1,"JUAN","PELAEZ","M",'1985-01-29',3500000);
    INSERT INTO EMPLEADO VALUES (2,"ANDRES","GARCIA","M",'1975-05-22',5500000);
    INSERT INTO EMPLEADO VALUES (3,"LAURA","PEREZ","F",'1991-09-10',2500000);
    INSERT INTO EMPLEADO VALUES (4,"PEPE","MARTINEZ","M",'1987-12-01',3800000);
    INSERT INTO EMPLEADO VALUES (5,"MARGARITA","CORRALES","F",'1990-07-02',4500000);

    INSERT INTO VACACIONES VALUES (1,1,'2019-07-01','2019-07-15','A',14);
    INSERT INTO VACACIONES VALUES (2,2,'2019-03-01','2019-03-15','R',14);
    INSERT INTO VACACIONES VALUES (3,2,'2019-04-01','2019-04-15','A',14);
    INSERT INTO VACACIONES VALUES (4,2,'2019-08-14','2019-08-20','A',6);
    INSERT INTO VACACIONES VALUES (5,3,'2019-08-20','2019-08-25','A',5);
    INSERT INTO VACACIONES VALUES (6,3,'2019-12-20','2019-12-31','A',11);
''')


# QUERY 1: Seleccione nombre, apellido y salario de todos los empleados.
query = '''
    SELECT NOMBRE, APELLIDO, SALARIO
    FROM EMPLEADO;
'''
query1_df = execute_query(query, cursor)

# QUERY 2: Seleccione nombre, apellido y salario de todos los empleados que ganen más de 4 millones.
query = '''
    SELECT NOMBRE, APELLIDO, SALARIO
    FROM EMPLEADO
    WHERE SALARIO > 4000000;
'''
query2_df = execute_query(query, cursor)

# QUERY 3: Cuente los empleados por sexo.
query = '''
    SELECT SEXO, COUNT(*) AS CANTIDAD
    FROM EMPLEADO
    GROUP BY SEXO;
'''
query3_df = execute_query(query, cursor)

# QUERY 4: Seleccione los empleados que no han hecho solicitud de vacaciones.
query = '''
    WITH EMP_VACACIONES AS (SELECT ID_EMP FROM VACACIONES)

    SELECT E.NOMBRE, E.APELLIDO
    FROM EMPLEADO AS E
    WHERE E.ID NOT IN EMP_VACACIONES;
'''
query4_df = execute_query(query, cursor)

# QUERY 5: Seleccione los empleados que tengan más de una solicitud de vacaciones y muestre cuantas solicitudes tienen los que cumplen.
query = '''
    SELECT E.NOMBRE, E.APELLIDO, COUNT(*) AS CNT_SOLICITUDES
    FROM EMPLEADO AS E
    INNER JOIN VACACIONES AS V 
    ON E.ID = V.ID_EMP
    GROUP BY E.ID
    HAVING COUNT(*) > 1;
'''
query5_df = execute_query(query, cursor)

# QUERY 6: Determine el salario promedio de los empleados.
query = '''
    SELECT AVG(SALARIO) AS SALARIO_PROMEDIO
    FROM EMPLEADO;
'''
query6_df = execute_query(query, cursor)

# QUERY 7: Determine la cantidad de días promedio solicitados de vacaciones por cada empleado.
query = '''
    SELECT E.NOMBRE, E.APELLIDO, AVG(V.CANTIDAD_DIAS) AS DIAS_PROMEDIO
    FROM EMPLEADO AS E
    INNER JOIN VACACIONES AS V 
    ON E.ID = V.ID_EMP
    GROUP BY E.ID;
'''
query7_df = execute_query(query, cursor)

# QUERY 8: Seleccione el empleado que mayor cantidad de días de vacaciones ha solicitado, muestre el nombre, apellido y cantidad de días totales solicitados.
query = '''
    SELECT E.NOMBRE, E.APELLIDO, SUM(V.CANTIDAD_DIAS) AS DIAS_TOTALES
    FROM EMPLEADO AS E
    INNER JOIN VACACIONES AS V 
    ON E.ID = V.ID_EMP
    GROUP BY E.ID
    ORDER BY SUM(V.CANTIDAD_DIAS) DESC
    LIMIT 1;
'''
query8_df = execute_query(query, cursor)

# QUERY 9: Consulte la cantidad de días aprobados y rechazados por cada empleado, en caso de no tener solicitudes mostrar 0.
query = '''
    SELECT E.NOMBRE, E.APELLIDO, 
    SUM(CASE WHEN V.ESTADO = 'A' THEN V.CANTIDAD_DIAS ELSE 0 END) AS APROBADOS,
    SUM(CASE WHEN V.ESTADO = 'R' THEN V.CANTIDAD_DIAS ELSE 0 END) AS RECHAZADOS
    FROM EMPLEADO AS E
    LEFT JOIN VACACIONES AS V 
    ON E.ID = V.ID_EMP
    GROUP BY E.ID;
'''
query9_df = execute_query(query, cursor)

# Export the dataframes querys to excel in different sheets
with pd.ExcelWriter('output/prueba_sql.xlsx') as writer:
    query1_df.to_excel(writer, sheet_name='query1')
    query2_df.to_excel(writer, sheet_name='query2')
    query3_df.to_excel(writer, sheet_name='query3')
    query4_df.to_excel(writer, sheet_name='query4')
    query5_df.to_excel(writer, sheet_name='query5')
    query6_df.to_excel(writer, sheet_name='query6')
    query7_df.to_excel(writer, sheet_name='query7')
    query8_df.to_excel(writer, sheet_name='query8')
    query9_df.to_excel(writer, sheet_name='query9')

# Commit the changes and close the connection
conn.commit()
conn.close()
