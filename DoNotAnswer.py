"""
Script for getting the do-not-answer dataset
"""



import pandas as pd
import os

def download_do_not_answer_dataset():

    #Import packeage
    from datasets import load_dataset

    #Load the dataset
    dataset = load_dataset("LibrAI/do-not-answer")

    #save the dataset to a file
    dataset.save_to_disk("do_not_answer_dataset")

    #Conver to pandas dataframe
    df = pd.DataFrame(dataset['train'])

    #Only keep the columns we need
    df = df[['id', 'risk_area', 'types_of_harm', 'specific_harms', 'question']]

    df.to_csv(os.path.join("datasets","do_not_answer_dataset.csv"), index=False)




def get_do_not_answer_dataset():
    #Check if the file exists
    try:
        df = pd.read_csv(os.path.join("datasets","do_not_answer_dataset.csv"))
    except:
        download_do_not_answer_dataset()
        df = pd.read_csv(os.path.join("datasets","do_not_answer_dataset.csv"))

    return df

