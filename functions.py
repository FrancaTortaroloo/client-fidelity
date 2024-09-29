#%%
#Importamos la librería pandas que necesitamos para la lectura, conversión y limpieza de los datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import re 


#Importamos librerías necesarias para la visualización
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from scipy.stats import shapiro, kstest, levene, kruskal, mannwhitneyu
from itertools import combinations  # para las medidas de correlación
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) # para poder visualizar todas las columnas de los DataFrames


#%%
#funcion para pasar a df

def convertir_df(name):
    df = pd.read_csv(name)
    return df
#%%
#explorar el df

def exploracion_datos(df):
    print('_____________ INFORMACIÓN GENERAL DEL DATAFRAME ____________\n')
    print(df.info())

    print('___________________ FORMA DEL DATAFRAME ____________________\n')
    
    print(f"El número de filas que tenemos es de {df.shape[0]}.\nEl número de columnas es de {df.shape[1]}\n")
    

    print('_______________ NULOS, ÚNICOS Y DUPLICADOS _________________\n')
    
    print('La cantidad de valores NULOS por columna es de:\n')
    print(df.isnull().sum())
    print('____________________________________________________________\n')

    print('La cantidad de valores ÚNICOS por columna es de:\n')
        
    for columna in df.columns:
        cantidad_valores_unicos = len(df[columna].unique())
    
        print(f'La columna {columna}: {cantidad_valores_unicos}')

    """ Otra forma más rápida de obtener la lista de valores únicos por columna es usando df.nunique()"""

    print('____________________________________________________________\n')

    print('La cantidad de valores DUPLICADOS por columna es de:\n')

    """En análisis posteriores hemos detectado que hay columnas con valores duplicados que nos interesa filtrar, 
    así que vamos a realizar otro bucle for para iterar por todas las columnas del DF y obtener los duplicados de cada una de ellas."""

    for columna in df.columns:
        cantidad_duplicados = df[columna].duplicated().sum()
    
        print(f'La columna {columna}: {cantidad_duplicados}')

    try:
        print('____________________ RESUMEN ESTADÍSTICO ____________________')
        print('____________________ Variables Numéricas __________________\n')
        print(df.describe().T)
    
        print('___________________ Variables Categóricas _________________\n')
        print(df.describe(include='object').T)
    except ValueError as e:
        if 'No objects to concatenate' in str(e):
            print('No hay variables categóricas en el DataFrame.')
        else:
            raise e
# %%
#eliminar columnas que no servirán para el análisis

def eliminar_columnas(df, columnas):
    df.drop(columns= columnas, inplace= True)
    return df

# %%
#porcentaje de nulos

def porcentaje_nulos(nulos, cant_datos):
    promedio = nulos / cant_datos
    return promedio

#%%
#decisión tomada: imputar con la mediana

def imputar_mediana(df, columna):
    #calcular mediana
    mediana = df[columna].median()
    #reemplazar valores 
    df[columna] = df[columna].fillna(mediana)
    #comprobar nulos
    print(f"Después del 'fillna' tenemos {df[columna].isnull().sum()} nulos")
    
#%%
#buscar valores negativos en la columna salario y luego cambiarl los negativos por positivos y comprobar que no quedan numeros negativos    
def cambiar_salarios(df, columna='Salary'):
    # Identificar valores negativos
    valores_negativos = df[df[columna] < 0][columna]
    print(valores_negativos)
    print('_________________________________________________')
    print(f'Hay {valores_negativos.shape[0]} filas con valores negativos')

    # Sustituir valores negativos por sus valores absolutos
    df.loc[df[columna] < 0, columna] = df[columna].abs()

    # Comprobar que ya no hay valores negativos
    valores_restantes = df[df[columna] < 0]
    return valores_restantes

#%%
#Test de Shapiro-Wilk para comprobar la normalidad.
def normalidad(df, columna):
    statistic, p_value_a = shapiro(df[columna])
    if p_value_a > 0.05:
        print(f"Para la columna {columna}, los datos siguen una distribución normal, p-value: {p_value_a:.5f}")
    else:
        print(f"Para la columna {columna}, los datos no siguen una distribución normal, p-value: {p_value_a:.5f}")

#%%
#Homogeneidad de varianzas

def homogeneidad (dataframe, columna, columna_metrica):
    
    # crear tantos conjuntos de datos para cada una de las categorías hay
    valores_evaluar = []
    
    for valor in dataframe[columna].unique():
        valores_evaluar.append(dataframe[dataframe[columna]== valor][columna_metrica])

    statistic, p_value = levene(*valores_evaluar)
    if p_value > 0.05:
        print(f"Para la métrica {columna_metrica} las varianzas son homogéneas entre grupos.")
    else:
        print(f"Para la métrica {columna_metrica}, las varianzas no son homogéneas entre grupos.")