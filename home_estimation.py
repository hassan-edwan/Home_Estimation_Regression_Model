import pandas as pd
import numpy as np

'''
create the data that the model will use to train
'''
def sort_data():
    redfin1 = pd.read_csv("redfin1.csv")
    redfin2 = pd.read_csv("redfin2.csv")
    redfin3 = pd.read_csv("redfin3.csv")
    redfin4 = pd.read_csv("redfin4.csv")
    redfin5 = pd.read_csv("redfin5.csv")
    redfin6 = pd.read_csv("redfin6.csv")
    redfin7 = pd.read_csv("redfin7.csv")
    redfin8 = pd.read_csv("redfin8.csv")
    redfin9 = pd.read_csv("redfin9.csv")
    redfin10 = pd.read_csv("redfin10.csv")
    redfin11 = pd.read_csv("redfin11.csv")
    redfin12 = pd.read_csv("redfin12.csv")
    housing_data = pd.concat([redfin1, redfin2, redfin3, redfin4, redfin5, redfin6, redfin7, redfin8, redfin9, redfin10, redfin11, redfin12])
    housing_data.drop_duplicates(subset = "ADDRESS", keep=False, inplace=True)




'''
input: model
output: expected value
create model
'''
def model_fit():
    pass

'''
train model
'''
def model_train():
    pass


'''
create a method of interacting with the model.
parameters: housing properties. Property Type, Zipcode
output: the expected value
'''

def iteraction():
    pass


def main():
    pass