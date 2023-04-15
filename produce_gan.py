import pandas as pd
import numpy as np
from ctgan import CTGANSynthesizer


def find_category(df):
    L=[]
    cols=df.columns
    i=0
    for col in cols:
        if len(df[col].unique())<=10:
            L.append(col)
            # print(col,len(df[col].unique()))
            i=i+1
    return L

def non_categorical(df,cat):
    cols = df.columns
    non_categorical=list(set(cols) - set(cat))
    return non_categorical



def run_ctgan(FilePath, rows):

    data = pd.read_csv(FilePath)
    
    discrete_columns=find_category(data)
    numeric=non_categorical(data,discrete_columns)


    ctgan = CTGANSynthesizer()
    ctgan.fit(data, discrete_columns, epochs=5)

    samples = ctgan.sample(rows)



    for cols in numeric:
        samples[cols]=samples[cols].astype(np.double)
        samples[cols]=abs(samples[cols].round())
    

    output_path='./Dataset/Gan_output.csv'
    output_path1='./Dataset/Gan_output_mixed.csv'
    samples.to_csv(output_path, index=False)

    df2=pd.read_csv(FilePath)
    df2['Synthetic Data']='No'
    samples['Synthetic Data']='Yes'
    gan = pd.concat([samples,df2])
    gan = gan.sample(frac = 1)
    gan.to_csv(output_path1, index=False)

    return output_path,output_path1



