from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import neighbors


def n_b (d_total, obj, t_size):
    (datos_entrenamiento, datos_val, obj_entrenamiento, obj_val) = model_selection.train_test_split(d_total, obj, test_size=t_size, random_state=2222)
    
    nbCategorical = naive_bayes.CategoricalNB()
    nbCategorical.fit(datos_entrenamiento, obj_entrenamiento)
    return round(nbCategorical.score(datos_val, obj_val), 3)


def r_lineal_multiple(d_total, obj, t_size):
    (datos_entrenamiento, datos_val, obj_entrenamiento, obj_val) = model_selection.train_test_split(
        d_total, obj, test_size=t_size, random_state=2222)

    lr_multiple = linear_model.LinearRegression()
    lr_multiple.fit(datos_entrenamiento, obj_entrenamiento)
    lr_predict = lr_multiple.predict(datos_val)
    y = 0
    for i in range(0, len(lr_predict)):
        rd = int(round(lr_predict[i], 0))
        if(rd == obj_val[i]):
            y = y + 1

    return round(y / len(obj_val), 3)

def r_logistica(d_total, obj, t_size, folds, cross):
    (datos_entrenamiento, datos_val, obj_entrenamiento, obj_val) = model_selection.train_test_split(d_total, obj, test_size=t_size, random_state=2222)
    if(cross == True):
        lg_regresion = linear_model.LogisticRegressionCV(cv=folds, random_state=0, solver='lbfgs', max_iter=1000)
        lg_regresion.fit(d_total,obj)
    else:
        lg_regresion = linear_model.LogisticRegression(penalty="none", solver='lbfgs', max_iter=1000)
        lg_regresion.fit(datos_entrenamiento,obj_entrenamiento)
    
    return round(lg_regresion.score(datos_val,obj_val),3)

def arboles_decision(d_total, obj, t_size):
    (datos_entrenamiento, datos_val, obj_entrenamiento, obj_val) = model_selection.train_test_split(d_total, obj, test_size=t_size, random_state=2222)
    
    arbol = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
    arbol.fit(datos_entrenamiento,obj_entrenamiento)
    return [round(arbol.score(datos_val,obj_val),3),arbol]


def knn(d_total,obj):
     for n in range(1,10):
        KNN= neighbors.KNeighborsClassifier(n_neighbors=n,metric="euclidean")
        KNN.fit(d_total,obj)
        scores = cross_val_score(KNN, X=d_total, y=obj, cv=10)
        media_score=0
        for i in scores:
            media_score+=i
        media_score/=scores.size

        return round(media_score,3)