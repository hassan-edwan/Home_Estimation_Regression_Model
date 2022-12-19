import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    ## drop the outliers
    df = df.loc[(df['PRICE'] <= 2000000) & (df['PRICE'] >= 20000)]
    df = df.loc[(df['BEDS'] <= 9)]
    
    

    ## standardize the length of each row in the zipcode column, wasnt registering leading zeros as part of the zip 
    df["ZIP_OR_POSTAL_CODE"] = df["ZIP_OR_POSTAL_CODE"].astype(str).str.zfill(5)
    ## only want the first 5 digits of each row, thats where the zip is
    df["ZIP_OR_POSTAL_CODE"] = df["ZIP_OR_POSTAL_CODE"].str[:5]
    df["ZIP_OR_POSTAL_CODE"]
    
    ## add new columns with location information based on zip code
    zips = df["ZIP_OR_POSTAL_CODE"]
    zips_index = zip_income["Zipcode"]
    states = df["STATE_OR_PROVINCE"]
    median_income_index = zip_income["Median Income "]
    growth_index = zip_income["Growth"]
    pop_size_index = zip_income["Population_size"]
    pop_age_index = zip_income["Population_age"]
    black_pop_index = zip_income["Percentage_Black"]
    immigrant_index = zip_income["Immigrant_pop"]
    pop_growth_index = zip_income["Pop_Growth"]

    # Create a lookup dictionary using the zip_index and median_income_index columns
    lookup = dict(zip(zips_index, median_income_index))
    lookup1 = dict(zip(zips_index, growth_index))
    lookup2 = dict(zip(zips_index, pop_size_index))
    lookup3 = dict(zip(zips_index, pop_age_index))
    lookup4 = dict(zip(zips_index, immigrant_index))
    lookup5 = dict(zip(zips_index, black_pop_index))
    #lookup6 = dict(zip(states_index, state_density_index))
    lookup7 = dict(zip(zips_index, pop_growth_index))
    
    
    # Use map to map the values in the zips column to the corresponding values in the median_income_index column using the lookup dictionary
    median_income = zips.map(lookup)
    growth = zips.map(lookup1)
    pop_size = zips.map(lookup2)
    pop_age = zips.map(lookup3)
    immigrants = zips.map(lookup4)
    black_pop = zips.map(lookup5)
    #state_density = states.map(lookup6)
    pop_growth = states.map(lookup7)
    
    # Replace any missing values with an appropriate value
    
    median_income = median_income.fillna(0)
    growth = growth.fillna(0)
    pop_size = pop_size.fillna(0)
    pop_age = pop_age.fillna(32)
    black_pop = black_pop.fillna(0)
    
    
    # Add the extra information columns to the df
    df['MEDIAN_INCOME'] = median_income
    df['GROWTH'] = growth
    df['POP_SIZE'] = pop_size
    df['POP_AGE'] = pop_age
    df["BLACK_POP"] = black_pop
    #df["DENSITY"] = state_density
    df["IMMIGRANTS"] = immigrants
    df["POP_GROWTH"] = pop_growth
    
    df = df.replace('-', 0)
    df = df. replace('#DIV/0!', 0)    
        
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
    
    
    df['STATE_OR_PROVINCE'] = df['STATE_OR_PROVINCE'].map(string_to_int_states)
    df['PROPERTY_TYPE'] = df['PROPERTY_TYPE'].map(string_to_int_property_type)
        
    # change all NaN elements to zero
    df.fillna(0, inplace=True)
    
    ## remove outliers by year
    df = df.loc[(df['YEAR_BUILT'] <= 2021) & (df['YEAR_BUILT'] >= 1900)]
    
    ## dataframe as csv file
    df.to_csv('housing_data.csv', index=False)
    return (df, zip_income) 


