## Potential Unit tests that I can make, accuracy of model unit test, if I make an interactive part then that too poten

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import streamlit as st

'''
create the data that the model will use to train
'''
def get_and_clean_data():
    ## read redfin data
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
    
    ## compile data into one dataframe
    housing_data = pd.concat([redfin1, redfin2, redfin3, redfin4, redfin5, redfin6,
                              redfin7, redfin8, redfin9, redfin10, redfin11, redfin12,
                              redfin13, redfin14, redfin15, redfin16, redfin17, redfin18,
                              redfin19, redfin20, redfin21, redfin22, redfin23, redfin24,
                              redfin25, redfin26, redfin27, redfin28])
    ## read income by zip data
    zip_income = pd.read_csv("income_by_zip.csv")
    ## get rid of leading characters in front of zip code
    zip_income['Zipcode'] = zip_income['Zipcode'].str[6:]
    ##dataframe as csv file, to make it easier to make sure I am making the changes I want to make
    zip_income.to_csv('zip_income.csv', index = False)
    ## get rid of addresses that repeat so that the same property isnt used multiple times
    housing_data.drop_duplicates(subset = "ADDRESS", keep=False, inplace=True)
    ## as dataframe, for some reason wasnt recognizing it as a dataframe
    df = pd.DataFrame(housing_data)
    ## get rid of spaces in the column headers, make it easier to call the columns
    df.columns = [c.replace(' ', '_') for c in df.columns]
    ## drop the data that I think is useless in determining the cost of the property or just gives it away ie: $/sqft
    df = df.drop(columns = ["SALE_TYPE", "SOLD_DATE", "ADDRESS", "LOCATION", "$/SQUARE_FEET", "HOA/MONTH", "STATUS", 
                       "NEXT_OPEN_HOUSE_START_TIME", "NEXT_OPEN_HOUSE_END_TIME", "URL_(SEE_https://www.redfin.com/buy-a-home/comparative-market-analysis_FOR_INFO_ON_PRICING)"
                       , "SOURCE", "MLS#", "FAVORITE", "INTERESTED"])
    ## standardize the length of each row in the zipcode column, wasnt registering leading zeros as part of the zip 
    df["ZIP_OR_POSTAL_CODE"] = df["ZIP_OR_POSTAL_CODE"].astype(str).str.zfill(5)
    ## only want the first 5 digits of each row, thats where the zip is
    df["ZIP_OR_POSTAL_CODE"] = df["ZIP_OR_POSTAL_CODE"].str[:5]
    df["ZIP_OR_POSTAL_CODE"]
    ## add new column with median income based on zip code
    
    zips = df["ZIP_OR_POSTAL_CODE"]
    zips_index = zip_income["Zipcode"]
    median_income_index = zip_income["Median Income "]
    growth_index = zip_income["Growth"]
    
    # Create a lookup dictionary using the zip_index and median_income_index columns
    lookup = dict(zip(zips_index, median_income_index))
    lookup1 = dict(zip(zips_index, growth_index))
    
    # Use map to map the values in the zips column to the corresponding values in the median_income_index column using the lookup dictionary
    median_income = zips.map(lookup)
    growth = zips.map(lookup1)
    
    # Replace any missing values with 0
    median_income = median_income.fillna(0)
    growth = growth.fillna(0)
    
    # Add the median_income values to the df
    df['MEDIAN_INCOME'] = median_income
    df['GROWTH'] = growth
    
    ## final adjustments to the data so that the model can run 
    string_to_int_property_type = {
        'Single Family Residential': 1,
        'Townhouse' : 2,
        'Condo/Co-op' : 3,
        'Multi-Family (2-4 Unit)' : 4,
        'Vacant Land': 5,
        'Mobile/Manufactured Home' : 6,
        'Multi-Family (5+ Unit)' : 7,
        'Ranch' : 8,
        'Co-op' : 9,
        'Timeshare' : 10,
        'Unknown' : 11,
        'Other' : 12,
        }
    string_to_int_states = {
        'NJ' : 1,
        'PA' : 2,
        'DE' : 3,
        'MD' : 4, 
        'AL' : 5,
        'GA' : 6,
        'FL' : 7,
        'VA' : 9,
        'NY' : 10,
        'MA' : 11,
        'CT' : 12,
        'RI' : 13,
        'NH' : 14,
        'VT' : 15,
        'DC' : 16,
        'WV' : 17,
        'OH' : 18,
        'NC' : 19,
        'TN' : 20,
        'KY' : 21,
        'IN' : 22,
        'MI' : 23,
        'IL' : 24,
        'WI' : 25,
        'IA' : 26,
        'SC' : 27
        }
    df.replace('00nan', 0, inplace = True)
    df.replace('00N/A', 0, inplace = True)
    
    df['STATE_OR_PROVINCE'] = df['STATE_OR_PROVINCE'].map(string_to_int_states)
    df['PROPERTY_TYPE'] = df['PROPERTY_TYPE'].map(string_to_int_property_type)
    
    # change all NaN elements to zero
    df.fillna(0, inplace=True)
    
    ## dataframe as csv file
    df.to_csv('housing_data.csv', index=False)
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
    ## data to show price by square footage
    plt.scatter(data.PRICE, data.SQUARE_FEET)
    plt.title("Price vs sqf Area")
    plt.xlim([0, 1000000])
    plt.ylim(0, 10000)
    plt.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='plain', axis='y', scilimits=(0,0))


'''
input: model
output: expected value
create model
'''
def create_model(data):
    ## Create Initial Regression Model:
    
    reg = LinearRegression()
    labels = data['PRICE']

    train1 = data.drop(['CITY', 'PRICE', 'DAYS_ON_MARKET'], axis = 1)
    # create training split
    x_train, x_test, y_train, y_test, = train_test_split(train1, labels, test_size = 0.10, random_state = 2)
    #fit the initial model
    reg.fit(x_train, y_train)
    
    ## Make Adjustments to model to make it more accurate:
    model = ensemble.GradientBoostingRegressor(n_estimators = 450, max_depth = 5, min_samples_split = 2,
                                             learning_rate= 0.1, loss = 'huber', max_features = "sqrt")
    ## fit the model
    model.fit(x_train, y_train)
    
    ## return the the model created
    return (model, x_train, y_train, x_test, y_test)


'''
create a method of interacting with the model.
parameters: housing properties. Property Type, State, Beds, Baths, Square footage, lot size, year built, lat, long 
output: the expected value
'''
## will add median income and growth later based on zip


def iteraction():
    property_types = ['Single Family Residential', 'Townhouse', 'Condo/Co-op', 'Multi-Family (2-4 Unit)', 'Vacant Land',
            'Mobile/Manufactured Home', 'Multi-Family (5+ Unit)', 'Ranch', 'Co-op', 'Timeshare', 'Unknown', 'Other']
    
    with st.echo(code_location = 'below'):
        st.slider = st.selectbox("What is the Property Type?", property_types, key = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


def main():
    data = get_and_clean_data()
    model = create_model(data)[0]
    x_train = create_model(data)[1]
    y_train = create_model(data)[2]
    x_test = create_model(data)[3]
    y_test = create_model(data)[4]
    new_input = []
    
    
if __name__ == "__main__":
    main()
    