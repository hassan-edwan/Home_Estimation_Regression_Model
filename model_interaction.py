import streamlit as st
import pandas as pd
from home_estimation import create_model
from redfin_data import get_and_clean_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from home_estimation import estimate_value
from home_estimation import get_model

def get_user_input(data):
    
    '''
    
    create a method of interacting with the model.
    parameters: housing properties. Property Type, State, Beds, Baths, Square footage, lot size, year built, lat, long 
    output: widgets to help the huser input the property parameters
    return the property parameters 
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
    
    ## get the median income and growth parameters Population_size,Population_age,Immigrant_pop,Num_Black_People,Percentage_Black,2012_pop_size,Pop_Growth
    zip_index = zipcodes['Zipcode'].tolist()
    median_index = zipcodes['Median Income '].tolist()
    growth_index = zipcodes['Growth'].tolist()
    pop_size_index = zipcodes['Population_size'].tolist()
    black_pop_index = zipcodes['Num_Black_People'].tolist()
    pop_age_index = zipcodes['Population_age'].tolist()
    immigrants_index = zipcodes['Immigrant_pop'].tolist()
    pop_growth_index = zipcodes['Pop_Growth'].tolist()
    
    properties_key = []
    for i in range(1, 13):
        properties_key.append(i)
    
    states_key = []
    for i in range(1, 50):
        states_key.append(i)

    ## add the widgets to the application
    
    st.caption("The coordinates of the properties that the model was trained on, it is clearly biased towards the north east corridor, and only encompasses the contiguous united states")
    st.map(coordinates)
    
    st.caption("Please Be Aware that the Location Parameters are not Locked to One Another. It is Possible to Input, Wildly Inaccurate Coordinates or Zipcode by State and Vice Versa")
    
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
    
    lat = st.number_input("what is the latitude of the property", min_value= -27.00, max_value = 47.00, value=35.00, step=1.000)
    st.write(f"You selected: {lat}")
    
    long = st.number_input("what is the longitude of the property", min_value= -120.00, max_value= -67.00, value=-80.00, step=1.000)
    st.write(f"You selected: {long}")
    
    st.write("coordinates of the property selected")
    st.map(pd.DataFrame({'lat': lat, 'lon': long}, index = [0]))
    
    for row in range(0, len(zipcodes)):
        if zipcode == int(zip_index[row]):
            median_income = median_index[row]
            growth = growth_index[row]
            pop_size = pop_size_index[row]
            black_pop = black_pop_index[row]
            pop_age = pop_age_index[row]
            immigrants = immigrants_index[row]
            pop_growth = pop_growth_index[row]
            break
        else:
            median_income = 0
            growth = 0
            pop_size = 0
            black_pop = 0
            pop_age = 0
            immigrants = 0
            pop_growth = 0
        if zipcode == int(zip_index[row]):
            median_income = median_index[row]
            growth = growth_index[row]
            pop_size = pop_size_index[row]
            pop_age = pop_age_index[row]
            black_pop = black_pop_index[row]
            immigrants = immigrants_index[row]
            pop_growth = pop_growth_index[row]
            break
        else:
            median_income = 0
            growth = 0
            pop_size = 0
            pop_age = 0
            black_pop = 0
            immigrants = 0
            pop_growth = 0
        
    property_type = string_to_int_property_type.get(property_type)
    state = string_to_int_states.get(state)
    ## PROPERTY_TYPE,STATE_OR_PROVINCE,ZIP_OR_POSTAL_CODE,PRICE,BEDS,BATHS,SQUARE_FEET,LOT_SIZE,YEAR_BUILT,
    ## LATITUDE,LONGITUDE,MEDIAN_INCOME,GROWTH,POP_SIZE,POP_AGE,BLACK_POP,IMMIGRANTS,POP_GROWTH
    
    new_input = [property_type, state, zipcode, num_beds, num_baths, square_feet,
                  lot_size, year_built, lat, long, median_income, growth, pop_size
                  , pop_age, black_pop, immigrants, pop_growth]
    
    return new_input
        
     
def output_to_user(new_input, model):
    
    
    st.write(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    new_output = estimate_value(model, new_input)
    st.metric("Predicted Value:", f"${int(new_output)}", "Model Accuracy: ~79.5%")
    


def main():
     data = get_and_clean_data()
     model = get_model()
     new_input = get_user_input(data)
     output_to_user(new_input, model)        

if __name__ == "__main__":
    main()
    