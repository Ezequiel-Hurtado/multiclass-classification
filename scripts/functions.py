import numpy as np 
import pandas as pd 
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

def categorical_corr(x,y):
    '''
    This variable calculates the classic Cramers V between two categorical 
    variables
    
    :param x: first categorical varible
    :param y: second categorical variable
    '''
    contingency_matrix = pd.crosstab(x,y)

    chi2_res = ss.chi2_contingency(contingency_matrix)[0]
    n = contingency_matrix.sum().sum()
    r, k = contingency_matrix.shape
    return np.sqrt(chi2_res/(n*(min(r,k)-1)))

def cat_corr_matrix(df): 
    cat_var = df.select_dtypes(include = 'object').columns
    cramer_matrix = pd.DataFrame(np.zeros((len(cat_var), len(cat_var))),
                                 index = cat_var, 
                                 columns= cat_var)
    
    for col1 in cat_var:
        for col2 in cat_var: 
            if col1 != col2:
                cramer_matrix.loc[col1,col2] = categorical_corr(df[col1],df[col2])
            else: 
                cramer_matrix.loc[col1,col2] = 1
    
    plt.figure(figsize = (12,8))
    sns.heatmap(cramer_matrix, annot = True, fmt = ".2f", cmap = 'coolwarm',
                center = 0 , vmin =0, vmax = 1)
    plt.title("Correlation Matrix for Categorical Variables (Cramers V)")
    plt.show()
    return

def continuous_cleaning(df_train, df_test,list): 
    for i in list: 
        median_value = df_train[i].median()
        df_train[i] = df_train[i].fillna(median_value)
        df_test[i] = df_test[i].fillna(median_value)

    df_train_res = df_train[df_train['N_Days'] <= df_train['Age']]
    
    return df_train_res[list].set_index(list[0]), df_test[list].set_index(list[0])

def categorical_cleaning(df_train, df_test):

    df_train['Drug'] = df_train['Drug'].where(df_train['Drug'].isin(['D-penicillamine', 'Placebo']),'Unknown')
    df_test['Drug'] = df_test['Drug'].where(df_test['Drug'].isin(['D-penicillamine', 'Placebo']),'Unknown')

    df_train['Sex'] = df_train['Sex'].where(df_train['Sex'].isin(['F', 'M']), df_train['Sex'].mode()[0])
    df_test['Sex'] = df_test['Sex'].where(df_test['Sex'].isin(['F', 'M']),df_train['Sex'].mode()[0])

    df_train['Ascites'] = df_train['Ascites'].where(df_train['Ascites'].isin(['Y', 'N']), df_train['Ascites'].mode()[0])
    df_test['Ascites'] = df_test['Ascites'].where(df_test['Ascites'].isin(['Y', 'N']),df_train['Ascites'].mode()[0])

    df_train['Hepatomegaly'] = df_train['Hepatomegaly'].where(df_train['Hepatomegaly'].isin(['Y', 'N']), 'Unknown')
    df_test['Hepatomegaly'] = df_test['Hepatomegaly'].where(df_test['Hepatomegaly'].isin(['Y', 'N']),'Unknown')

    df_train['Spiders'] = df_train['Spiders'].where(df_train['Spiders'].isin(['Y', 'N']), df_train['Spiders'].mode()[0])
    df_test['Spiders'] = df_test['Spiders'].where(df_test['Spiders'].isin(['Y', 'N']),df_train['Spiders'].mode()[0])
    
    df_train['Edema'] = df_train['Edema'].where(df_train['Edema'].isin(['Y', 'N','S']), df_train['Edema'].mode()[0])
    df_test['Edema'] = df_test['Edema'].where(df_test['Edema'].isin(['Y', 'N','S']),df_train['Edema'].mode()[0])

    return  df_train, df_test
