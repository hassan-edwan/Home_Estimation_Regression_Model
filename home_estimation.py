import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
create the data that the model will use to train
'''
def get_data():

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
    redfin13 = pd.read_csv("redfin13.csv")
    redfin14 = pd.read_csv("redfin14.csv")
    redfin15 = pd.read_csv("redfin15.csv")
    redfin16 = pd.read_csv("redfin16.csv")
    redfin17 = pd.read_csv("redfin17.csv")
    redfin18 = pd.read_csv("redfin18.csv")
    redfin19 = pd.read_csv("redfin19.csv")
    redfin20 = pd.read_csv("redfin20.csv")
    redfin21 = pd.read_csv("redfin21.csv")
    redfin22 = pd.read_csv("redfin22.csv")
    redfin23 = pd.read_csv("redfin23.csv")
    redfin24 = pd.read_csv("redfin24.csv")
    redfin25 = pd.read_csv("redfin25.csv")
    redfin26 = pd.read_csv("redfin26.csv")
    redfin27 = pd.read_csv("redfin27.csv")
    redfin28 = pd.read_csv("redfin28.csv")
    
    housing_data = pd.concat([redfin1, redfin2, redfin3, redfin4, redfin5, redfin6,
                              redfin7, redfin8, redfin9, redfin10, redfin11, redfin12,
                              redfin13, redfin14, redfin15, redfin16, redfin17, redfin18,
                              redfin19, redfin20, redfin21, redfin22, redfin23, redfin24,
                              redfin25, redfin26, redfin27, redfin28])
    housing_data.drop_duplicates(subset = "ADDRESS", keep=False, inplace=True)
    df = pd.DataFrame(housing_data)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    return df

"""
inspect data

"""
def inspect_data(data):
    print(data.describe())
    ## make graph based on bedroom count data
    data["BEDS"].value_counts().plot(kind='bar')
    plt.title('number of rooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Count')
    sns.despine
    ## make graph based on cost to longitude data
    plt.scatter(data.PRICE,data.LONGITUDE)
    plt.title('cost of house by rooms')
    plt.xlabel('price')
    plt.ylabel('long.')
    plt.show()
    ## data to show location of houses by long and lat
    plt.figure(figsize = (10, 10))
    sns.jointplot(x=data.LONGITUDE.values, y=data.LATITUDE.values, size = 10)
    plt.ylabel("longitude")
    plt.xlabel("latitiude")
    plt.show()
    sns.despine
    


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
    data = get_data()
    print(data)
    plt.scatter(data.PRICE, data.SQUARE_FEET)
    plt.title("Price vs sqf Area")
    plt.xlim([0, 1000000])
    plt.ylim(0, 10000)
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
if __name__ == "__main__":
    main()