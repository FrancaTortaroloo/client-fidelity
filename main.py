#%%
import functions as func

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, levene, kruskal, mannwhitneyu
#%% 
print('Fase 1: Exploración y limpieza')

#%%
df_flight = func.convertir_df('Customer Flight Activity.csv')
df_loyalty = func.convertir_df('Customer Loyalty History.csv')

#%%
df_flight
#%%
df_loyalty
#%%
#exploraremos lo que hay dentro de cada df
explorar_flight = func.exploracion_datos(df_flight)

#%%
explorar_loyalty = func.exploracion_datos(df_loyalty)
# %%
#unir ambos dataframes

merge = df_flight.merge(df_loyalty, left_on= 'Loyalty Number', right_on = 'Loyalty Number' )
merge

#%%
#elminar duplicados

merge.drop_duplicates()

# %%
#eliminar columnas que no servirán para el análisis

columnas_eliminar= ['Flights with Companions', 'Points Redeemed', 'Dollar Cost Points Redeemed','Postal Code', 'CLV', 'Enrollment Type', 'Enrollment Month', 'Cancellation Year', 'Cancellation Month' ]

eliminar = func.eliminar_columnas(merge, columnas_eliminar)
eliminar
# %%

#confirmar que se han eliminado las columnas

func.exploracion_datos(merge)


#%% 
#Recursos para analizar con qué conviene imputar
#Boxxplot para ver la distribucion de outliers y la mediana

plt.figure(figsize=(5, 2.5))
sns.boxplot(x=merge['Salary'])
plt.title('Boxplot Salary')
plt.xlabel('Valor')
plt.show()

# %%
#comprobar la media y la mediana para ver si hay mucha diferencia entre sí

print(f'La media es de {round(merge["Salary"].mean(), 2)}')
print(f'La mediana es de {round(merge["Salary"].median(), 2)}')
# %%
#cantidad de nulos en la columna salario
nulos_salario = merge['Salary'].isnull().sum()
nulos_salario
#%%
func.porcentaje_nulos(nulos_salario, 405624)
# %%
#imputar con la mediana y confirmar que no hay nulos

func.imputar_mediana(merge, 'Salary')
# %%
#comprobar si hay valores negativos, cambiar de negarivos a positivos
valores_restantes = func.cambiar_salarios(merge)
# %%
#Comprobamos que ya no hay valores negativos
merge[merge['Salary'] < 0]

# %%
print('Fase 2: Visualización')
#%%
print('¿Cómo se distribuye a cantidad de vuelos reservados por mes durante el año?')

#comprobar los años que hay en la columna Year
merge['Year'].unique()

#%% 
#contar cantidad de vuelos por mes y año
grouped_df = merge.groupby(['Year','Month'])['Flights Booked'].sum().reset_index()
grouped_df

#%%
#graficar 
# Crear el gráfico usando barplot

plt.figure(figsize=(12, 6))


sns.barplot(x="Month", 
            y="Flights Booked", 
            hue="Year", 
            data=grouped_df,
            palette="pastel", 
            edgecolor="black",
            width = 0.8)

# Añadir títulos y etiquetas
plt.title('Cantidad de vuelos reservados por mes durante los años 2017 y 2018', fontsize=20)
plt.xlabel('Mes', fontsize=15)
plt.ylabel('Cantidad de vuelos reservados', fontsize=15)

# Mostrar gráfico
'''plt.show()


print('Existe una relación entre la distancia de los vuelos y los puntos acumulados por los clientes?')

plt.bar(x= merge['Points Accumulated'], height= merge['Distance'])
plt.xlabel("Puntos Acumulados")
plt.ylabel("Distancia");
'''
#%%
#Lo muestro en un mapa de calor para ver mejor la correlación

# Seleccionar solo las columnas que queremos correlacionar
df_correlacion = merge[['Points Accumulated', 'Distance' ]]

# Calcular la matriz de correlación
matriz_correlacion = df_correlacion.corr()

# Crear el mapa de calor
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# título del mapa de calor
plt.title('Mapa de Calor de la Correlación entre Puntos Acumulados y Distancia de los vuelos')

# Mostrar el mapa de calor
plt.show()

#%%
print('Se puede decir que sí existe una relación entre la distancia de los uelos y los puntos acumulados por los clientes')

#%%
print('Cuál es la distribución de los clientes por provincia?')

#Agrupar para facilitar la visualizacion

grouped_df = merge.groupby('Province')['Loyalty Number'].count().reset_index()
grouped_df
# %%
# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(grouped_df['Province'], grouped_df['Loyalty Number'], color='skyblue')

# Añadir títulos y etiquetas
plt.title('Distribución de clientes por provincia')
plt.xlabel('Provincia')
plt.ylabel('Cantidad de clientes')

# Mejorar la legibilidad de las etiquetas del eje x
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.tight_layout()
plt.show()
#%%
print('Se puede notar que la mayor concentración de clientes está en Ontario, seguido por la provincia de British Columbia y por último Quebec')

#%%
print('¿Cómo se compara el salario promedio entre los diferentes niveles educativos de los clientes?')

#Agrupar para facilitar la visualización

grouped_educacion = merge.groupby('Education')['Salary'].mean().reset_index()
grouped_educacion
# %%
# Crear el gráfico de barras
plt.figure(figsize=(16, 8))
plt.bar(grouped_educacion['Education'], grouped_educacion['Salary'], color='skyblue')

# Añadir títulos y etiquetas
plt.title('Salario promedio entre los niveles educativos de clientes', fontsize=20)
plt.xlabel('Nivel de educación', fontsize=20)
plt.ylabel('Salario promedio', fontsize=20)


# Mostrar el gráfico
plt.tight_layout()
plt.show()

#%%
print('Se puede notar que las personas que tienen un doctorado tienen en promedio un salario más alto, seguido por los que han hecho un máster')
# %%
print('¿Cuál es la proporción de clientes con diferentes tipos de tarjetas de fidelidad?')

#Agrupar para facilitar la visualizacion

grouped_loyalty = merge.groupby('Loyalty Card')['Loyalty Number'].count().reset_index()
grouped_loyalty
# %%
# Crear el gráfico de barras
plt.figure(figsize=(16, 8))
plt.bar(grouped_loyalty['Loyalty Card'], grouped_loyalty['Loyalty Number'], color='skyblue')

# Añadir títulos y etiquetas
plt.title('Proporción de clientes con tipo de tarjetas de fidelidad', fontsize=20)
plt.xlabel('Tipo de tarjeta', fontsize=20)
plt.ylabel('Cantidad de clientes', fontsize=20)


# Mostrar el gráfico
plt.tight_layout()
plt.show()
# %%
print('La mayor cantidad de clientes se concentra en la tarjeta de fidelidad Star, mientras que Aurora es la que menos afiliados tiene')

#%%
print('¿Cómo se distribuyen los clientes según su estado civil y género?')

sns.countplot(x = "Marital Status", 
              data = merge,
              hue = 'Gender', 
              palette = "pastel",
              edgecolor = 'black'
              );

# Añadir títulos
plt.title('Distribución de clientes según su estado civil y género', fontsize=20)
# %%
print('La mayoría de los clientes son casados, indistintamente del género')
#%%
print('Fase 3: Evaluación de diferencias en las reservas de vuelos por nivel educativo')
#%%
#me quedo sólo con las columnas que necesito en un df nuevo

df_nuevo = merge[['Flights Booked', 'Education']]
df_nuevo

#%%
# Crear columnas nuevas y hacer el conteo de 'Flights Booked' por nivel educativo
df_agrupado = df_nuevo.groupby('Education').agg({'Flights Booked': 'count'}).reset_index()
df_agrupado.columns = ['Education', 'Count Flights Booked']
df_agrupado

# %%
#Análisis descriptivo
df_analisis = df_nuevo.groupby('Education')['Flights Booked'].describe().T
df_analisis

# %%
print('Prueba estadística: \n Hipótesis nula: No existe diferencia significativa entre los niveles educativos \n Hipótesis alternativa: Existe diferencia siginificativa en el número de vuelos reservados entre los diferentes niveles educativos.')
# %%
#Test de Shapiro-Wilk para comprobar la normalidad.
metricas = ['Bachelor', 'College', 'Doctor', 'High School or Below', 'Master']

for metrica in metricas:
    func.normalidad(df_analisis, metrica)
# %%
print('Comprobar la homogeneidad de varianzas')
#%%
df_nuevo['Bachelor'] = df_nuevo[df_nuevo['Education'] == 'Bachelor']['Flights Booked']
df_nuevo['College'] = df_nuevo[df_nuevo['Education'] == 'College']['Flights Booked']
df_nuevo['Doctor'] = df_nuevo[df_nuevo['Education'] == 'Doctor']['Flights Booked']
df_nuevo['High School or Below'] = df_nuevo[df_nuevo['Education'] == 'High School or Below']['Flights Booked']
df_nuevo['Master'] = df_nuevo[df_nuevo['Education'] == 'Master']['Flights Booked']

grupos = ['Bachelor', 'Doctor', 'Master', 'College', 'High School or Below']

#Itero sobre cada grupo y llamo a la función homogeneidad.
for grupo in grupos:
    func.homogeneidad(df_nuevo, 'Education', grupo)
# %%
print('He elegido utilizar el test de Kruskal-Wallis debido a que son datos numéricos, con una distribución que no es normal y porque se comparan un total de 5 grupos.')
#%%
# Aplicamos la prueba de Kruskal-Wallis
statistic, p_value = kruskal(
    df_analisis['Bachelor'],
    df_analisis['Doctor'],
    df_analisis['Master'],
    df_analisis['College'],
    df_analisis['High School or Below']
)

# Imprimimos los resultados
print(f"Estadístico de prueba de Kruskal-Wallis: {statistic}")
print(f"Valor p: {p_value}")

# Comparamos el valor p con el nivel de significancia
alfa = 0.05
if p_value > alfa:
    print("No podemos rechazar la hipótesis nula.")
    print("No hay diferencias significativas entre los grupos.")
else:
    print("Rechazamos la hipótesis nula.")
    print("Hay diferencias significativas entre al menos un par de grupos.")

# %%
print('Conclusión: por el análisis estadístico de los datos (la media de cada grupo y las pruebas estadísticas), no existe una diferencia significativa en el número de vuelos reservados entre los diferentes niveles educativos.')
# %%
