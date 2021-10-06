# %%
from numpy.lib.function_base import copy
import pandas as pd
import networkx as nx
from funciones import n_b,r_lineal_multiple,r_logistica,arboles_decision,knn
from sklearn import tree
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt


#Leemos los datos y le indicamos que no tenemos cabecera, quitamos la primero fila y columna que son  indices
matrix = pd.read_csv("CSV/LONDON_GANG.csv",header=None)
matrix = matrix.iloc[1: , 1:]

#Comprobamos que esta correctamente guardado
print(matrix.head())


atributos = pd.read_csv("CSV/LONDON_GANG_ATTR.csv",header=0)
atributos = atributos.iloc[:, 1:]

print(atributos.head())

#Creamos un grafo con la matriz de adyacencia y comprobamos que se ha creado correctamente, en nuestro caso lo dibujamos numerado
grafo = nx.Graph(matrix.values)
#nx.draw_networkx(grafo, node_size=130)


#Vamos a obtener los atributos relacionales, vienen como un diccionario asi que cogemos los valores y los ponemos en forma de lista
centralidad_grado = list(nx.degree_centrality(grafo).values())
centralidad_intermedia = list(nx.betweenness_centrality(grafo).values())
centralidad_katz = list(nx.katz_centrality_numpy(grafo, alpha=0.1, beta=1.0).values())

print("CENTRALIDAD DE GRADO\n")
print(centralidad_grado)
print("\n====================================================================================================================================================\n")
print("CENTRALIDAD DE INTERMEDIACIÓN\n")
print(centralidad_intermedia)
print("\n====================================================================================================================================================\n")
print("CENTRALIDAD DE KATZ\n")
print(centralidad_katz)
print("\n====================================================================================================================================================\n")

#Los grados vienen datos en tuplas con (indice,grado) de cada nodo, en nuestro caso solo  remos el grado
grados = [j for i,j in grafo.degree]
print(grados)
print("\n====================================================================================================================================================\n")

#Al igual que los anteriores metodos, devuelve un diccionario y pasamos los valores a una lista
clusters = list(nx.clustering(grafo).values())
print(clusters)
print("\n====================================================================================================================================================\n")

'''
#Coloreado
coloreado = [j for i,j in sorted(nx.greedy_color(grafo).items())]
print(coloreado)
print("\n====================================================================================================================================================\n")
'''

atributos_relacionales = {"Centralidad_grado":centralidad_grado, "Centralidad_intermedia":centralidad_intermedia, 
                          "Centralidad_katz":centralidad_katz, "Grados":grados, "Clusters":clusters}

#Creamos los dataframes con los nuevos datos juntos y separados
atributos_total = atributos.copy()
for k,v in atributos_relacionales.items():
    atributos_total[k] = v
    
datos = []    
for k,v in atributos_relacionales.items():
    atributos_aux = atributos.copy()
    atributos_aux[k] = v
    datos.append(atributos_aux)

#Datos a usar
t_size = 0.25
at_objetivo = "Prison"
objetivo = atributos[[at_objetivo]].values.ravel()
atributos.drop(axis=1, columns = [at_objetivo], inplace=True)
atributos_total.drop(axis=1, columns = [at_objetivo], inplace=True)

for i in range(0,len(datos)):
    datos[i].drop(axis=1, columns = [at_objetivo,"Music"], inplace=True)

print("Se ha usado un porcetaje de entrenamiento de ", (1.0-t_size) * 100,"%\n")

#Naive Bayes
print("\nNAIVE BAYES\n")

print(f"Acierto con atributos originales para {at_objetivo} : {n_b(atributos, objetivo, t_size)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {n_b(atributos_total, objetivo, t_size)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",n_b(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[6]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",n_b(dat_usado, objetivo, t_size),"\n")


#Regresion Lineal Multiple
print("\nRegresión Lineal\n")

print(f"Acierto con atributos originales para {at_objetivo} : {r_lineal_multiple(atributos, objetivo, t_size)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {r_lineal_multiple(atributos_total, objetivo, t_size)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",r_lineal_multiple(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[6]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",r_lineal_multiple(dat_usado, objetivo, t_size),"\n")

#Regresión Logística
print("\nRegresión Logistica Test-Split\n")


print(f"Acierto con atributos originales para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 0, False)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 0, False)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",r_logistica(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size, 0, False),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[6]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",r_logistica(dat_usado, objetivo, t_size, 0, False),"\n")


print("\nRegresión Logistica Cross-Validation\n")

print(f"Acierto con atributos originales para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 10, True)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 10, True)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",r_logistica(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size, 10, True),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[6]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",r_logistica(dat_usado, objetivo, t_size, 10, True),"\n") 


#Árboles de decisión
print("\nArboles de decisión\n")

print(f"Acierto con atributos originales para {at_objetivo} : {arboles_decision(atributos, objetivo, t_size)[0]}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {arboles_decision(atributos, objetivo, t_size)[0]}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",arboles_decision(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size)[0],"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[6]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",arboles_decision(dat_usado, objetivo, t_size)[0],"\n")
  


#KNN
print("\nKNN - Euclidea\n")

print(f"Acierto con atributos originales para {at_objetivo} : {knn(atributos_total, objetivo)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {knn(atributos_total, objetivo)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",knn(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[6]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",knn(dat_usado, objetivo),"\n")
    


'''Nos hemos percatado de que a pesar que el consumir música puede parecer trivial, hemos decido cercionarnos de que tanto peso aproximadamente 
podria tener dicho atributo en los datos mediante los coeficientes de decision. Podemos observar que Music tiene un peso considerable, si bien retirar el atributo 
mejora algunos resultados en la mayoria de casos los empeora'''

(datos_entrenamiento, datos_val, obj_entrenamiento, obj_val) = model_selection.train_test_split(atributos_total, objetivo, test_size=t_size, random_state=2222)

lg_regresion = linear_model.LogisticRegression(solver='lbfgs',max_iter=1000, multi_class='multinomial')
lg_regresion.fit(datos_entrenamiento,obj_entrenamiento)
coefs = lg_regresion.coef_.tolist()[0]
atr = atributos_total.columns.tolist()
d = dict(zip(atr,coefs))

for k,v in d.items():
    print(k,round(v,4))

d = dict(d)
plt.bar(range(len(d)), list(d.values()), align='center')
plt.xticks(range(len(d)), list(d.keys()),rotation='vertical')
plt.show()
    
    
