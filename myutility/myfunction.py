import pandas as pd
import numpy as np

def myTrainTestSplit(dataframe, index_train, index_test, targetVarName, dropna=False):
    df_train = dataframe.loc[index_train]
    df_test  = dataframe.loc[index_test]
    if dropna:        
        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)
    
    ## séparation des variables d'entrées et variable cible
    X_train = df_train.drop(targetVarName, axis=1)
    X_test  =  df_test.drop(targetVarName, axis=1) 
    
    y_train = df_train[targetVarName]
    y_test  =  df_test[targetVarName]
    
    return X_train, X_test, y_train, y_test




def get_table_line(modelName, scores, paramToDisplay=None, modelparams = None ):
    '''
    - but : 
    afficher sur une même ligne de tableau les performance d'un modèle
    'nom modèle' : 'des hyperparamètres' : 'des scores'
    
    - attributs : 
    modelName (str): nom du modèle à afficher
    modelparams (dict): l'ensemble des hyperparamètres du modèle {nom : valeur}
    score (dataframe): résultat du modèle sur la validation croisé (cross_validate())
    paramToDisplay (list): spécifie les hyperparamètres du modèle à afficher
    '''
    
    output = []
    output.append(modelName)
    if paramToDisplay:
        for param in paramToDisplay:
            if param not in modelparams:
                output.append('-')
                continue    ## si le params à afficher n'existe pas, on passe
            if type(modelparams[param]) != str: ## si la valeur du param. est un nombre
                if modelparams[param] < 1e4:
                    output.append(round(modelparams[param],2))
                else:
                    output.append('%.2E' % round(modelparams[param],2)) ## notation scientifique
            else:
                output.append(modelparams[param])
        
    scores = scores.describe()
    output.append(round(scores['test_score']['mean'], 3))
    output.append(round(scores['test_score']['std'], 3))
    output.append(round(scores['fit_time']['mean'], 2))
    return output


def get_best_params(paramsModel, df_results):
    qry = ' and '.join(['param_{} == {}'.format(k,v) for k,v in paramsModel.items()])
    selection = df_results.query(qry).sort_values(by='rank_test_score').iloc[0]
    return selection.params


def add_test_score(dfOutput, scoresCV, scoreEval, suffixMod = ''):
    '''
    entrées :
    - dfOutput : df auquel on ajoute horizontalement les mse des modèles
    - scoresCV (dict) dictionnaire contenant le nom du modèle et le score (résultat de CrossValidate)
    - scoresEval (dict) dictionnaire contant le nom du modèle et la MSE calculé sur jeu d'évaluation
    
    sortie :
    - dfOutput
    '''

    for key in scoresCV:
        df_cv = scoresCV[key][['test_score']]
        modelName = key + suffixMod
        df_cv.rename(columns= {'test_score' : modelName}, inplace=True)
        df_eval = pd.DataFrame({modelName: [scoreEval[key]]})
        df_concat = pd.concat([df_cv, df_eval])
        dfOutput = pd.concat([dfOutput, df_concat], axis = 1)
        
    return dfOutput


def star_line_table(score, energyStar, model, encoding):
    output = []
    destmp = score.describe()
    output.append(energyStar)
    output.append(model)
    output.append(encoding)
    output.append(round(destmp.loc['mean','test_score'],3))
    output.append(round(destmp.loc['std','test_score'],3))
    output.append(round(destmp.loc['mean','fit_time'],3))
    
    return output



def convertToTargetMoment(df, target):
    targetMean = []   ## moyenne
    targetVar  = []   ## variance
    targetSkew = []   ## skewness
    targetKurt = []   ## kurtosis
    
    inputCateg = df.select_dtypes('object').columns.to_list()
    
    ## pour chaque variable catégorielle, on crée son encodage Target
    for cat in inputCateg:
        newNameMean = cat + '_meanEncoded'
        newNameVar  = cat + '_varEncoded'
        newNameSkew = cat + '_skewEncoded'
        newNameKurt = cat + '_kurtEncoded'
        
        ## associe à chaque modalité -> mean(y)
        encodingMean =  df.groupby(cat)[target].mean()
        encodingVar  =  df.groupby(cat)[target].var()
        encodingSkew =  df.groupby(cat)[target].skew()
        encodingKurt =  df.groupby(cat)[target].apply(pd.DataFrame.kurt)
    #df.groupby('a').apply(pd.DataFrame.kurt)
        
        ## crée colonne de la variable categ. encodée 
        df[newNameMean] = df[cat].map(encodingMean)
        df[newNameVar]  = df[cat].map(encodingVar)
        df[newNameSkew] = df[cat].map(encodingSkew)
        df[newNameKurt] = df[cat].map(encodingKurt)
    
        targetMean.append(newNameMean)
        targetVar.append(newNameVar)
        targetSkew.append(newNameSkew)
        targetKurt.append(newNameKurt)
                
    return df.drop(inputCateg, axis='columns')


def gathercateg(dataframe, threshold=0.05, cumulative=False, dropna=True):

    listCategVar = dataframe.dtypes == 'object'
    listCategVar = listCategVar[listCategVar].index.tolist()
    
    if cumulative:
        for categVar in listCategVar:
        
            smallCategories = dataframe[categVar].value_counts(normalize=True, ascending=True, dropna=dropna).cumsum() < threshold 
            
            otherfreq = 0.
            if 'Other' in dataframe[categVar].unique():
                otherfreq = dataframe[categVar].value_counts(normalize=True)['Other']
                if not smallCategories['Other']:  ## we add to the last (X - Other) % to the Other category
                    smallCategories = dataframe[categVar].value_counts(normalize=True, ascending=True, dropna=dropna).cumsum() < threshold - otherfreq 
            
            smallCategories = smallCategories[smallCategories]
            dataframe.replace( {categVar : dict.fromkeys(smallCategories.index, 'Other')}, inplace=True)
            
    else:
        for categVar in listCategVar:
           
            smallCategories = dataframe[categVar].value_counts(normalize=True, ascending=True, dropna=dropna) < threshold 
            smallCategories = smallCategories[smallCategories]
            dataframe.replace( {categVar : dict.fromkeys(smallCategories.index, 'Other')}, inplace=True)


