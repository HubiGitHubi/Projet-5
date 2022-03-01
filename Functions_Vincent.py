import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from datetime import date
#from functions import *
import seaborn  as sns
from sklearn import metrics

def plot_val_util_parligne(Df):
    colonne_tocheck = Df.columns

    taux_na_per_ligne = (100-(Df.isnull().sum(axis=1)/len(Df.columns) * 100)).sort_values()
    plt.title('Taux de valeurs utilisables par ligne')

    plt.plot(taux_na_per_ligne.values)
    plt.show()

def info_data(Df,Df1) : 
    """Compare les données entre deux Dataframe
    retourne les type de variables,nblignes, nbcolonnes, Taux de valeurs utilisables
    Df : Le Df originel
    Df1 : Le Df modifié
    """
    print("les colonnes initiales sont de types :  ",Df.dtypes.unique())
    print("les colonnes sont maintenant de types : ",Df1.dtypes.unique())
    print("  ")
    print("il y a intitialement :",len(Df),"lignes,",len(Df.columns),"colonnes",)
    print("il y a maintenant :   ",len(Df1),"lignes,",len(Df1.columns),"colonnes",)
    print("  ")
    print("il y a intitialement :", round(100- (Df.isna().sum().sum()/np.size(Df)*100),1),"% de valeurs utilisables")
    print("il y a maintenant :   ", round(100- (Df1.isna().sum().sum()/np.size(Df1)*100),1),"% de valeurs utilisables")

def info_data_unique(Df) : 
    """
    retourne les type de variables,nblignes, nbcolonnes, Taux de valeurs utilisables
    Df : Le Df 

    """
    print("les colonnes sont de types :  ",Df.dtypes.unique())
    print("  ")
    print("il y a :",len(Df),"lignes,",len(Df.columns),"colonnes",)
    print("  ")
    print("il y a :", round(100- (Df.isna().sum().sum()/np.size(Df)*100),1),"% de valeurs utilisables")
    
def Utilisables_par_col(Df):
    """ 
    renvoie le taux de valeurs utilisables  pour chaque colonne
    input : Dataframe à regrouper
    retourne le taux de valeur utilisable
    """
    util_percol = 100-(Df.isna().sum()/len(Df)*100).sort_values(ascending=False)
    return util_percol

def plot_valeurs_utilisables(Df) :
    """ 
    plot le taux de valeur utilisable 
    input : Df à ploter
    """
    util_per_col = Utilisables_par_col(Df)
    fig1, ax = plt.subplots(figsize=(35, 10))
    ax.grid(True, which='both')
    #Traçons le taux de valeurs utilisables par colonnes
    plt.bar(util_per_col.index,  util_per_col.values)
    plt.title("Données utilisables par colonnes (%)",fontsize=25)
    plt.ylabel("Données utilisables par colonnes (%)",fontsize=25)
    plt.xticks(fontsize=20,rotation = 90)
    plt.yticks(fontsize=20)
    plt.grid(True) 
    plt.show()
    
def time_to_datetime(Df,colonne) :
    """passe les valeurs du format time au format Datetime
    input : dataFrame, et la colonne à passer en Datetime 
    Return : Datframe avec la colonne choisit au format Datetime
    """
    Df[colonne] = pd.to_datetime([d for d in Df[colonne] ], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    return Df

def object_to_string(Df):
    """ Convertit les colonnes de types objets en type string
    input : le Df à convertir
    return le Df convertit
    """
    ColDf = Df.columns[Df.dtypes=='object']
    Df[ColDf]= Df[ColDf].astype('string')
    return Df


                  
def plot_comparaison_valeurs_utilisables(Gb_Indicator_name,Gb_Indicator_name1) :
    """ 
    plot le taux de valeur utilisable des 2 Df l'un à coté de l'autre
    input : Df à ploter,Df1 à comparer avec Df1
    """
    df = pd.DataFrame({'Données brutes': Gb_Indicator_name,'Après suppr/Modif': Gb_Indicator_name1},\
                      index=Gb_Indicator_name1.index)
    
                 
    ax = df.plot.bar(rot=0,figsize=(35, 10),fontsize=25)
    plt.title("Données utilisables par colonnes (%)",fontsize=25)
    plt.xticks(fontsize=20,rotation = 90)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=30,loc= 'upper right')
    plt.grid(True)
    
def plotboxDf(Df,Colonnes_names,activate_list_of_filters):
    
    """Plot un plotbox pour chaque colonne de colonnes_names
    input : Le Df à ploter, les colonnes à ploter,
    activate_list_of_filters  = 1 : plot seulement les colonnes contenues entre 0 et 100g
    activate_list_of_filters  = 0: plot toutes les colonnes
    """
    
    if activate_list_of_filters == 1 :
        liste_of_filters = ['energy_100g','nutrition-score-fr_100g','index']
        
        for Filter in liste_of_filters :
            Colonnes_names = Colonnes_names[~(Colonnes_names.str.contains(Filter)) ]
    else :
        liste_of_filters = []
                  
    xtick = [x for x in range(1,len(Colonnes_names)+1) ]
    xtickNames = Colonnes_names
    fig2,ax = plt.subplots(figsize=(20, 5))
    plt.boxplot(Df[Colonnes_names].dropna())
    plt.title("valeurs de tous les produits, par variables")

    #plt.boxplot(Df[Filter_100g(Df,liste_of_filters)].dropna())

    plt.xticks(xtick, xtickNames,rotation=45)
    plt.grid(True)   
    plt.show()    
    
    
def Verif_col1_inferieur_col2(Df,liste_col_Big,liste_col_small) :
    """vérifie si  chaque valeur de Df[liste_col_Big] <  à chaque valeur Df[liste_col_small], 
    sinon remplace les valeurs Df[liste_col_small]>Df[liste_col_Big] par None
    input : Le Dataframe avec les valeurs à vérifier, 2 listes de la forme :
    [col_Big1, col_Big2 ,col_Big3 ]col_small3,...]
    [col_small1, col_small2, col_small3,...]
    
    Attention : liste_col_Big,liste_col_small doivent être appairée comme ceci :
    col_Big1 appairée avec col_small1 etc...
        
    Return le Df originel avec les valeurs aberantes remplacées par None
    """
    for nb_col in range(len(liste_col_small)) :
        Df[liste_col_small[nb_col]][Df[liste_col_small[nb_col]] > Df[liste_col_Big[nb_col]]] = None
        
    return Df

def Filter_min_Na(Df,Taux_valeur_min) :
    """
    Filtre les colonnes de "Gb_Indicator" dont le taux de valeur utilisable est inférieur à "Taux_valeur_min"
    retourne le 
    input : DataFrame à filter
    le taux à filter entre 0 et 100 
    (0 = aucune valeur utilisables, 100 = 100% de valeurs utilisables)
    retourne List_indic, la liste des colonnes filtrées
    """
    util_per_col = Utilisables_par_col(Df)    
    indic_search = [util_per_col.index[util_per_col.values>Taux_valeur_min]]
    List_indic = []
    for indic in indic_search[0] :
        List_indic.append(indic)
        
    return List_indic


def verif_max_Na_perligne_100g(Df,colonne_tocheck,Taux_Na_Max) :
    """ Renvoie les index du DF avec moins que Taux_Na_Max valeurs Nan
    input : Df, les colonnes à vérifier, le taux de Nan Max choisit
    Taux_Na_Max : 0 aucune valeur utilisables, 100 : 100% de valeur utilisables
    return: la liste d'index 
    """
    index_percent_missing = Df[Df[colonne_tocheck].isnull().sum(axis=1) \
                         * 100 / len(Df[colonne_tocheck].columns)<Taux_Na_Max].index
    return index_percent_missing

def Kdeplot_filter100g(Df,Df1) :
    """Kdeplot pour comparer 2 Df pour les colonnes contenant des valeurs numériques
    input : Le Df et Df1
    """
    colonnes= Df.select_dtypes(include=['float64']).columns
    for x in colonnes :
        fig3 = plt.figure()
        plt.subplot(2, 1, 1)
        sns.kdeplot(data=Df[[x]], x=x , label = x)
        sns.kdeplot(data=Df1[[x]], x=x , label = x)
        plt.legend(["Df before imputer ","Df after imputer "])
        plt.grid(True)
        plt.show()    
        
def ShowErrorCol1inCol2(Df,col_supposed_smallest, col_supposed_biggest) :
    """plot les 50 1eres valeurs (Max) aberrantes (col_supposed_smallest > col_supposed_biggest) s'il yen a 
    input : Le Df à vérifier, la colonne sencée êtr eplus petite, la colonne sencée être plus grande
    return un plot des des 50 1eres valeurs aberrantes s'il yen a
    """
  
    Df_fat_error = Df[[col_supposed_biggest,col_supposed_smallest]][Df[col_supposed_smallest] >Df[col_supposed_biggest]]
    Df_fat_error = Df_fat_error.sort_values(by=col_supposed_smallest,ascending = False)[0:50]

    if Df_fat_error.empty :
        print("Aucune valeur incohérente entre les colonnes",col_supposed_smallest," et ",col_supposed_biggest)
    else :   
        #Traçons le taux de valeurs utilisables par colones            
        ax = Df_fat_error.plot.bar(rot=0,figsize=(35, 10),fontsize=25)
        plt.title("Valeurs incohérentes : Col_small>Col_big",fontsize=25)
        plt.xticks(fontsize=20,rotation = 90)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=30)
        plt.grid(True)
        
def scatter_100g(Df, Liste_colonne) : 
    """Plotons les valeurs de chaque colonnes contenants des valeurs numériques
    input : Dataframe à ploter, les colonnes choisis
    """
    fig,ax = plt.subplots(figsize=(8,4))

    for c in Liste_colonne:
        ax.scatter( [c]*len(Df), Df[c], s=8)
    
    ax.tick_params(axis='x' , rotation=65)  
    ax.set_title("valeurs de tous les produits, par variables")
def scatter_resultat(Df,nom_method):

    list_model = list(Df[nom_method])
    y_test = Df.iloc[:,0]
    sizes = {} # clé : coordonnées ; valeur : nombre de points à ces coordonnées
    for (yt, yp) in zip(list(y_test), list_model):
        if (yt, yp) in sizes:
            sizes[(yt, yp)] += 1
        else:
            sizes[(yt, yp)] = 1

    keys = sizes.keys()
    plt.scatter(
            [k[0] for k in keys], # vraie valeur (abscisse)
            [k[1] for k in keys], # valeur predite (ordonnee)
            s=[sizes[k] for k in keys], # taille du marqueur
            color='coral', alpha =0.8)
    plt.xlabel("y_test")
    plt.ylabel(nom_method)
    plt.title("y_test en fonction de y_pred avec la méthode "+str(nom_method))

    plt.show()
    
def scatter_comp_2variables(df, list_2_cols) :
    plt.scatter(df[list_2_cols[0]],df[list_2_cols[1]])
    plt.xlabel(list_2_cols[0])
    plt.ylabel(list_2_cols[1])
    plt.title(str(list_2_cols[0])+ " en fonction de "+ str(list_2_cols[1]) )
    
def result_after_training(X_train_std,X_test_std,y_train,y_test,y_cols,model,Df_error,gridCV=None):
    """Fonction permettant de :
    - entrainer le model choisit( avec un gridsearchcv si choisit) 
    -Calculer y_pred pour chaque colonne de Y
    -printer les meilleurs paramètres de gridsearchcv s'il yen a un
    -ajouter une ligne au Df_error avec, le nom de la methode,la variable à prédire, la RMSE et l'erreur moyenne
    
    """
    best_param = 1
    
    if gridCV==None:
        gridCV = model
        best_param = 0
        
    for col in [0,1]:
        # On entraîne ce modèle sur les données d'entrainement
        gridCV.fit(X_train_std,y_train[:, col])

        #calcul des prédictions
        y_pred = gridCV.predict(X_test_std)

            

        #erreur sur les valeurs de test
        error = np.mean((y_pred - y_test[:, col]) ** 2)
        error_square = np.sqrt(metrics.mean_squared_error(y_test[:, col], y_pred))

        #erreur sur les valeurs de train
        error_train = np.mean((gridCV.predict(X_train_std) - y_train[:, col]) ** 2)
        error_square_train = np.sqrt(metrics.mean_squared_error(gridCV.predict(X_train_std), y_train[:, col]))

        print(model)
        print("feature à prédire : ",y_cols[col])
        print("l'erreur quadratique minimum est sur le test:", error_square)
        print("l'erreur quadratique minimum est sur le train:", error_square_train)
        print("l'erreur minimum est sur le test:", error)
        print("l'erreur minimum est sur le train:",error_train )
        print("  \n")
        if best_param == 1 :
            print("best_params :", gridCV.best_params_)
        print("  \n")    
        plt.scatter(y_test[:, col],y_pred)
        plt.title('    Scatter test / pred avec la méthode '+ str(model),loc='right')
        plt.xlabel(y_cols[col]+ '_test')
        plt.ylabel(y_cols[col]+ '_pred')
        plt.legend(y_cols[col:])
        plt.show()
    
        #Remplissons le Df error
        Df_error.loc[len(Df_error)+1] = [str(model),
                                        y_cols[col],
                                        error_square,
                                            error]
    return Df_error