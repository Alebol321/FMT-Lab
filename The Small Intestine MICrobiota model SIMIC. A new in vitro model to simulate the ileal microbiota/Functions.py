# LIbrary used!

import matplotlib.pyplot as plt # graphs
import pandas as pd # dataframes
import numpy as np #math is mathing
import seaborn as sns # nice graphs and statistics
import re # for regular expression management
import os #for directory management 
from sklearn.metrics import r2_score #to calculate the r2 score of linear regressions
import scipy.stats
from sklearn.preprocessing import StandardScaler # in order to standardize my data for the PCA
from sklearn.decomposition import PCA # in order to do PCA
import sys
# multiple tests
from scipy.stats import ttest_ind
from scipy.stats import shapiro 
from scipy.stats import lognorm
from scipy.stats import bartlett
from scipy.stats import mannwhitneyu
from scipy.stats import levene
import starbars # putting start in graphs
import mne.stats
from scipy.spatial.distance import cosine #cosine distance
from matplotlib.gridspec import GridSpec #making graph better

######################################
# Functioned defined
#######################################
def standard_curve(file,bacteria):
    """takes in a file containing data for a standard curve in a plate and makes the linear regression in order to make the analysis later, it will only do one bacteria at the time"""
    plt.close()
    df=pd.read_excel(file)
    df2=df[df["Bacteria"]==bacteria][["Ct","Copies"]]
    print(df2)
    x=df2.Ct
    y=df2.Copies
    idx = np.isfinite(x) & np.isfinite(y) # unfortunately polyfit is stupid and doesn´t ignore NaN by itself, so we have to tell it that there are
    model = np.polyfit(x[idx], y[idx], 1) #creates the linear regression, the one is for the grade of the polynomial
    print("The linear regression has angular coefficient of "+str(model[0])+" and intercept of "+str(model[1]))
    predict = np.poly1d(model)
    print("The r2 score of our linear model is "+ str(r2_score(y[idx],predict(x[idx]))))
    # we will also plot it
    x_lin_reg = range(int(np.min(x)-3), int(np.max(x))+3)
    y_lin_reg = predict(x_lin_reg)
    #plt.scatter(x,y) #creates the plot with just the points
    #plt.plot(x_lin_reg, y_lin_reg, c = 'r')
    #plt.title("{0} standard curve".format(bacteria))
    #plt.xlabel("initial copy number (log10)")
    #plt.ylabel("ct")
    #plt.show()
    #print(predict)
    return predict


def plate_analysis(file,big_df,file_standardcurve,sheet_name=False):
    """takes in data from a plate, and uses the standard curve in order to obtain the real value for bacteria concentration and puts everything together in a big dataframe"""
    if sheet_name==False:
        df=pd.read_excel(file)
    else:
        df=pd.read_excel(file,sheet_name=sheet_name)
    df=df[["Bacteria","Time","Ct"]]
    df=df[np.isfinite(df["Ct"])] #removes the columns that have NaNs
    print(df)
    for el in df["Bacteria"].unique():#per ogni batterio diverso nella piastra, leggiamo la standard curve
        print(el)
        df2=df[df["Bacteria"]==el]
        predict=standard_curve(file_standardcurve,el)
        for i in df2.index:
            big_df.loc[len(big_df.index)] = [el,df2.loc[i].iloc[1],predict(df2.loc[i].iloc[2])]# puts inside the big dataframe lines with the absolute value already normalized on the standard curve, the condition, and the bacteria


def plotter(big_df,title=None,perturbation=None):
    """takes a dataframe with ct, condition, bacteria as columns and does the lineplot"""
    plt.close()
    fig, axes = plt.subplots(1, 3,figsize=(25,8))
    palette=sns.color_palette("bright")
    palette1=palette[0:3]
    palette2=palette[3:5]
    palette3=palette[6:9]
    sns.lineplot(data=big_df[(big_df["Bacteria"].isin(["E.faecalis","S.lutetiensis","S.salivarius"])) ], x="Time",y="Copies", hue="Bacteria",hue_order=["E.faecalis","S.lutetiensis","S.salivarius"],marker="o",linestyle="--", linewidth=1.5,palette=palette1,ax=axes[0],err_style='bars')
    sns.lineplot(big_df[(big_df["Bacteria"].isin(["B.adolescentis","B.fragilis"]))], x="Time",y="Copies", hue="Bacteria",hue_order=["B.adolescentis","B.fragilis"],linestyle="--",marker="o", linewidth=1.5,palette=palette2,ax=axes[1],err_style='bars')
    sns.lineplot(big_df[(big_df["Bacteria"].isin(["B.wadsworthia","V.parvula","R.hominis"]))], x="Time",y="Copies", hue="Bacteria",hue_order=["B.wadsworthia","V.parvula","R.hominis",],linestyle="--",marker="o", linewidth=1.5,palette=palette3,ax=axes[2],err_style='bars')
    # setting axes labels, nd ticks
    fig.suptitle(title,size=20)
    for i in range(3):
        axes[i].set_xlabel("Time (h)",size=12)
        axes[i].set_ylabel("Amplicon copies (log10) ",size=12)
        axes[i].set_yticks(range(1,10))
        axes[i].legend(prop={'style': 'italic',"size":15},loc="lower right")
    if perturbation != None:
        for i in range(len(perturbation)): # number of perturbation present
            for j in range(3):
                axes[j].axvline(x=perturbation[i],color="red",linestyle="dotted") 


def relative_abundance(dataf):
    ### takes in a dataframe with columns TIme, Bacteria, and Copies and returns a data frame with the relative abundances + a plot of them
    new_df=pd.DataFrame()
    new_df = pd.DataFrame(columns=['Time'])  # Start with a Time column
    
    for _, row in dataf.iterrows():
        time_point = row["Time"]
        bacteria = row["Bacteria"]
        copies = 10**row["Copies"]
    
        if bacteria not in new_df.columns:
            new_df[bacteria] = None  # Add bacteria column if not present
    
        # Check if the time point exists
        if time_point in new_df["Time"].values:
            idx = new_df.index[new_df["Time"] == time_point][0]  # Get index of the existing time row
            if pd.isna(new_df.at[idx, bacteria]):  # If it's NaN, assign directly
                new_df.at[idx, bacteria] = copies
            else:
                # If already has a value, compute the average
                new_df.at[idx, bacteria] = (new_df.at[idx, bacteria] + copies) / 2
        else:
            # Append new row
            new_row = pd.Series([time_point] + [None] * (len(new_df.columns) - 1), index=new_df.columns)
            new_row[bacteria] = copies
            new_df = pd.concat([new_df, new_row.to_frame().T], ignore_index=True)
    # Ensure numeric columns are properly converted
    new_df.iloc[:, 1:] = new_df.iloc[:, 1:].apply(pd.to_numeric)
    new_df=new_df[new_df["Time"]!=0]
      plt.legend(loc="lower left")
    return new_df


def modify_df(df):
    ### Takes dataframe with Bacteria time and copies as columns and return a df with time as index and different bacteria as columns
    new_df=pd.DataFrame()
    new_df = pd.DataFrame(columns=['Time'])  # Start with a Time column
    for _, row in df.iterrows():
        time_point = row["Time"]
        bacteria = row["Bacteria"]
        copies = row["Copies"]
        if bacteria not in new_df.columns:
            new_df[bacteria] = None  # Add bacteria column if not present
        # Check if the time point exists
        if time_point in new_df["Time"].values:
            idx = new_df.index[new_df["Time"] == time_point][0]  # Get index of the existing time row
            if pd.isna(new_df.at[idx, bacteria]):  # If it's NaN, assign directly
                new_df.at[idx, bacteria] = copies
            else:
                # If already has a value, compute the average
                new_df.at[idx, bacteria] = (new_df.at[idx, bacteria] + copies) / 2
        else:
            # Append new row
            new_row = pd.Series([time_point] + [None] * (len(new_df.columns) - 1), index=new_df.columns)
            new_row[bacteria] = copies
            new_df = pd.concat([new_df, new_row.to_frame().T], ignore_index=True)
    # Ensure numeric columns are properly converted
    new_df.iloc[:, 1:] = new_df.iloc[:, 1:].apply(pd.to_numeric)
    new_df=new_df[new_df["Time"]!=0]
    new_df.set_index("Time",inplace=True)
    return new_df


#####################
#Ecological parameters function
########################

def canberra (v1, v2):
    ### canberra distance between 2 vectors, it is assumed the 2 vector are the same length
    sum=0
    for i in range(len(v1)):
        if v1.iloc[i]+v2.iloc[i]!=0: # case in which both are 0
            distance= abs(v1.iloc[i]-v2.iloc[i])/(abs(v1.iloc[i])+abs(v2.iloc[i]))
            sum+=distance
    return sum/len(v1)   



def euclidean (v1,v2):
    ### euclidean distance between 2 vectors, it is assumed the 2 vector are the same length
    s=0
    normalizer=0 # we need a normalizing factor to have a value between 0 and 1
    for i in range(len(v1)):
        distance=np.square(v1.iloc[i]-v2.iloc[i])
        s+=distance
        maximum = max(v1.iloc[i],v2.iloc[i])
        normalizer+=np.square(maximum)
    if normalizer == 0:
        return 0
    else:
        return (np.sqrt(s)/np.sqrt(normalizer)) #needed to have a correct scale between 1 and 0




def canberra_matrix (df):
    if 6 not in df.columns:
        df=df.T
    df2 = pd.DataFrame(index=df.columns,columns=df.columns)
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            print(df[df.columns[i]])
            print(df[df.columns[j]])
            df2.iloc[i,j] = canberra(df[df.columns[i]],df[df.columns[j]])
    return df2




def euclidean_matrix (df):
    if 6 not in df.columns:
        df=df.T
    df2 = pd.DataFrame(index=df.columns,columns=df.columns)
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            df2.iloc[i,j] = euclidean(df[df.columns[i]],df[df.columns[j]])
    return df2




def cosine_matrix(df):
    """from a group of vectors, calculates the cosine distance between all of them"""
    if 6 not in df.columns:
        df=df.T
    df2 = pd.DataFrame(index=df.columns,columns=df.columns)
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            df2.iloc[i,j] = cosine(df[df.columns[i]],df[df.columns[j]])
    return df2
