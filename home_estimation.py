## Potential Unit tests that I can make, accuracy of model unit test, if I make an interactive part then that too poten

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import streamlit as st


def get_and_clean_data():
    '''
    create the data that the model will use to train
    '''
    
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
    redfin29 = pd.read_csv("redfin29.csv")
    redfin30 = pd.read_csv("redfin30.csv")
    redfin31 = pd.read_csv("redfin31.csv")
    redfin32 = pd.read_csv("redfin32.csv")
    redfin33 = pd.read_csv("redfin33.csv")
    redfin34 = pd.read_csv("redfin34.csv")
    redfin35 = pd.read_csv("redfin35.csv")
    redfin36 = pd.read_csv("redfin36.csv")
    redfin37 = pd.read_csv("redfin37.csv")
    redfin38 = pd.read_csv("redfin38.csv")
    redfin39 = pd.read_csv("redfin39.csv")
    redfin40 = pd.read_csv("redfin40.csv")
    redfin41 = pd.read_csv("redfin41.csv")
    redfin42 = pd.read_csv("redfin42.csv")
    redfin43 = pd.read_csv("redfin43.csv")
    redfin44 = pd.read_csv("redfin44.csv")
    redfin45 = pd.read_csv("redfin45.csv")
    redfin46 = pd.read_csv("redfin46.csv")
    redfin47 = pd.read_csv("redfin47.csv")
    redfin48 = pd.read_csv("redfin48.csv")
    redfin49 = pd.read_csv("redfin49.csv")
    redfin50 = pd.read_csv("redfin50.csv")
    redfin51 = pd.read_csv("redfin51.csv")
    redfin52 = pd.read_csv("redfin52.csv")
    redfin53 = pd.read_csv("redfin53.csv")
    redfin54 = pd.read_csv("redfin54.csv")
    redfin55 = pd.read_csv("redfin55.csv")
    redfin56 = pd.read_csv("redfin56.csv")
    redfin57 = pd.read_csv("redfin57.csv")
    redfin58 = pd.read_csv("redfin58.csv")
    redfin59 = pd.read_csv("redfin59.csv")
    redfin60 = pd.read_csv("redfin60.csv")
    redfin61 = pd.read_csv("redfin61.csv")
    redfin62 = pd.read_csv("redfin62.csv")
    redfin63 = pd.read_csv("redfin63.csv")
    redfin64 = pd.read_csv("redfin64.csv")
    redfin65 = pd.read_csv("redfin65.csv")
    redfin66 = pd.read_csv("redfin66.csv")
    redfin67 = pd.read_csv("redfin67.csv")
    redfin68 = pd.read_csv("redfin68.csv")
    redfin69 = pd.read_csv("redfin69.csv")
    redfin70 = pd.read_csv("redfin70.csv")
    redfin71 = pd.read_csv("redfin71.csv")
    redfin72 = pd.read_csv("redfin72.csv")
    
    
    ## compile data into one dataframe
    housing_data = pd.concat([redfin1, redfin2, redfin3, redfin4, redfin5, redfin6,
                              redfin7, redfin8, redfin9, redfin10, redfin11, redfin12,
                              redfin13, redfin14, redfin15, redfin16, redfin17, redfin18,
                              redfin19, redfin20, redfin21, redfin22, redfin23, redfin24,
                              redfin25, redfin26, redfin27, redfin28, redfin29, redfin30,
                              redfin31, redfin32, redfin33, redfin34, redfin35, redfin36,
                              redfin37, redfin38, redfin39, redfin40, redfin41, redfin42,
                              redfin43, redfin44, redfin45, redfin46, redfin47, redfin48,
                              redfin49, redfin50, redfin51, redfin52, redfin53, redfin54,
                              redfin55, redfin56, redfin57, redfin58, redfin59, redfin60,
                              redfin61, redfin62, redfin63, redfin64, redfin65, redfin66,
                              redfin67, redfin68, redfin69, redfin70, redfin71, redfin72])
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
    'NJ': 1,
    'PA': 2,
    'DE': 3,
    'MD': 4,
    'AL': 5,
    'GA': 6,
    'FL': 7,
    'VA': 8,
    'NY': 9,
    'MA': 10,
    'CT': 11,
    'RI': 12,
    'NH': 13,
    'VT': 14,
    'DC': 15,
    'WV': 16,
    'OH': 17,
    'NC': 18,
    'TN': 19,
    'KY': 20,
    'IN': 21,
    'MI': 22,
    'IL': 23,
    'WI': 24,
    'IA': 25,
    'SC': 26,
    'MN': 27,
    'KS': 28,
    'MO': 29,
    'NE': 30,
    'SD': 31,
    'ND': 32,
    'MT': 33,
    'WY': 34,
    'CO': 35,
    'NM': 36,
    'AZ': 37,
    'UT': 38,
    'NV': 39,
    'ID': 40,
    'OR': 41,
    'WA': 42,
    'CA': 43,
    'HI': 44,
    'AK': 45,
    'ME': 46,
    'OK': 47,
    'AR': 48,
    'LA': 49,
    'MS': 50,
    'TX': 51,
    }
    df.replace('00nan', 0, inplace = True)
    df.replace('00N/A', 0, inplace = True)
    
    print(df['PROPERTY_TYPE'].nunique())
    
    df['STATE_OR_PROVINCE'] = df['STATE_OR_PROVINCE'].map(string_to_int_states)
    df['PROPERTY_TYPE'] = df['PROPERTY_TYPE'].map(string_to_int_property_type)
    
    # change all NaN elements to zero
    df.fillna(0, inplace=True)
    
    
    ## dataframe as csv file
    df.to_csv('housing_data.csv', index=False)
    return (df, zip_income) 

def inspect_data(data):
    
    
    """
    inspect data
    
    """
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


def create_model(data):
    '''
    input: model
    output: expected value
    create model
    '''

    ## Create Initial Regression Model:
    
    reg = LinearRegression()
    labels = data['PRICE']

    train1 = data.drop(['CITY', 'PRICE', 'DAYS_ON_MARKET'], axis = 1)
    # create training split
    x_train, x_test, y_train, y_test, = train_test_split(train1, labels, test_size = 0.10, random_state = 3)
    #fit the initial model
    reg.fit(x_train, y_train)
    
    ## Make Adjustments to model to make it more accurate:
    model = ensemble.GradientBoostingRegressor(n_estimators = 500, max_depth = 5, min_samples_split = 2,
                                             learning_rate= 0.1, loss = 'huber', max_features = "sqrt")
    
    
    ## fit the model
    model.fit(x_train, y_train)
    
    ## check accuracy
    accuracy = model.score(x_test, y_test)
    
    ## return the the model created
    return (model, accuracy)

"""
def interaction(data, model, accuracy):
    
    '''
    create a method of interacting with the model.
    parameters: housing properties. Property Type, State, Beds, Baths, Square footage, lot size, year built, lat, long 
    output: the expected value
    '''
    zipcodes = data[1]
    full_data = data[0]
    
    latitude = full_data['LATITUDE'].tolist()
    longitude = full_data['LONGITUDE'].tolist()
    lat_long = {"latitude": latitude, "longitude": longitude}
    coordinates = pd.DataFrame(lat_long)
    
    ## will add median income and growth later based on zip
    
    property_types = ['Single Family Residential', 'Townhouse',
            'Condo/Co-op', 'Multi-Family (2-4 Unit)', 'Vacant Land',
            'Mobile/Manufactured Home', 'Multi-Family (5+ Unit)',
            'Ranch', 'Co-op', 'Timeshare', 'Unknown', 'Other']
    states = [
            'NJ', 'PA', 'DE', 'MD', 'AL', 'GA', 'FL', 'VA',
            'NY', 'MA', 'CT', 'RI', 'NH', 'VT', 'DC', 'WV',
            'OH', 'NC', 'TN', 'KY', 'IN', 'MI', 'IL', 'WI',
            'IA', 'SC']
    
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
    'NJ': 1,
    'PA': 2,
    'DE': 3,
    'MD': 4,
    'AL': 5,
    'GA': 6,
    'FL': 7,
    'VA': 8,
    'NY': 9,
    'MA': 10,
    'CT': 11,
    'RI': 12,
    'NH': 13,
    'VT': 14,
    'DC': 15,
    'WV': 16,
    'OH': 17,
    'NC': 18,
    'TN': 19,
    'KY': 20,
    'IN': 21,
    'MI': 22,
    'IL': 23,
    'WI': 24,
    'IA': 25,
    'SC': 26,
    'MN': 27,
    'KS': 28,
    'MO': 29,
    'NE': 30,
    'SD': 31,
    'ND': 32,
    'MT': 33,
    'WY': 34,
    'CO': 35,
    'NM': 36,
    'AZ': 37,
    'UT': 38,
    'NV': 39,
    'ID': 40,
    'OR': 41,
    'WA': 42,
    'CA': 43,
    'HI': 44,
    'AK': 45,
    'ME': 46,
    'OK': 47,
    'AR': 48,
    'LA': 49,
    'MS': 50,
    'TX': 51,
    }
    
    ## get the median income and growth parameters
    zip_index = zipcodes['Zipcode'].tolist()
    median_index = zipcodes['Median Income '].tolist()
    growth_index = zipcodes['Growth'].tolist()
           
    
    properties_key = []
    for i in range(1, 13):
        properties_key.append(i)
    
    states_key = []
    for i in range(1, 27):
        states_key.append(i)

    ## add the widgets to the application
    with st.echo(code_location = 'below'):
        
        st.write("The coordinates of the properties that the model was trained on, it is clearly biased towards the north east corridor, and only encompasses parts of the eastern third of the us, which means it will be less accurate outside of those areas")
        st.map(coordinates)
        
        property_type = st.selectbox("What is the Property Type?", property_types, key = properties_key)
        st.write("You selected: ", property_type)
        
        state = st.selectbox("What State is the Property in?", states, key = states_key)
        st.write("You selected: ", state)
        
        num_beds = st.number_input("How many Bedrooms does the Property have", min_value=0, max_value=10, value=3, step=1)
        st.write(f"You selected: {num_beds}")
        
        num_baths = st.number_input("How many Bathrooms does the Property have", min_value=0.0, max_value=8.0, value=2.0, step=0.5)
        st.write(f"You selected: {num_baths}")
        
        square_feet = st.number_input("How many Square Feet does the Property have", min_value=100, max_value=6000, value=1500, step=1)
        st.write(f"You selected: {square_feet}")
        
        lot_size = st.number_input("How big is the lot size", min_value=0, max_value=1000000, value=2500, step=1)
        st.write(f"You selected: {lot_size}")
        
        year_built = st.number_input("What year was the property built in", min_value= 1930, max_value=2021, value=1990, step=1)
        st.write(f"You selected: {year_built}")
        
        zipcode =  st.number_input("What zipcode was the property built in", min_value = 601, max_value = 99929, value=20850, step=1)
        st.write(f"You selected: {zipcode}")
        
        lat = st.number_input("what is the latitude of the property", min_value= -27.00, max_value = 47.00, value=35.00, step=1.00)
        st.write(f"You selected: {lat}")
        
        long = st.number_input("what is the longitude of the property", min_value= -120.00, max_value= -67.00, value=-80.00, step=1.00)
        st.write(f"You selected: {long}")
        
        st.write("coordinates of the property selected")
        st.map(pd.DataFrame({'lat': lat, 'lon': long}, index = [0]))
        
        for row in range(0, len(zipcodes)):
            if zipcode == int(zip_index[row]):
                median_income = median_index[row]
                growth = growth_index[row]
                break
            else:
                median_income = 0
                growth = 0
                
        property_type = string_to_int_property_type.get(property_type)
        state = string_to_int_states.get(state)
        
        new_input = [property_type, state, zipcode, num_beds, num_baths, square_feet,
                      lot_size, year_built, lat, long, median_income, growth]
        new_input_array = np.array(new_input).reshape(1, -1)

        new_output = model.predict(new_input_array)
        col1, col2 = st.columns(2)
        col1.metric("Predicted Value:", f"${int(new_output)}")
        col2.metric("Model Accuracy:", accuracy)
"""
def main():
    
    ## get data that will be used
    data = get_and_clean_data()
    ## get model data
    model = create_model(data[0])[0]
    accuracy = create_model(data[0])[1]
    
    print(accuracy)
    
    ##return the interactive portion and get the user inputted variables
    ##print(interaction(data, model, accuracy))

    
if __name__ == "__main__":
    main()
    