#Import dependencies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as mms

#Import My custom modules
import src.wrangle_zillow as wr
import src.env as env


#Define a query for the mall dataset
mall_query = '''
             SELECT *
             FROM customers
             '''


#Create the url to access the database
database = 'mall_customers'


#pulls the mall database from the codeup database and returns a dataframe
def acquire_mall():    

    url = env.get_db_url(database)
    df =  pd.read_sql(mall_query, url)

    return df 
    

# returns a histogram of a distribution of the spending score for mall customers    
def describe_mall(df):
    plt.hist(df['spending_score'])
    

# 
def fix_columns(df):
    df = pd.get_dummies(df)
    df.drop(columns=['gender_Male'], inplace=True)
    
    return df


# removes values above the 95th percentile and below the 5th percentile
def remove_outliers(df):
    age_upper = df['age'].quantile(.95)
    age_lower = df['age'].quantile(.05)

    df = df[df['age'] <= age_upper]
    df = df[df['age'] > age_lower]

    return df 


def scale_data(train):
    columns= ['age', 'annual_income']
    scaler = mms()
    scaler.fit(train[columns])
    train[columns] = scaler.transform(train[columns])
    return train


def split(df):
    #Split dataframe into three subsets
    seed = 7

    train, validate = train_test_split(df, train_size=0.7,
                                    random_state=seed)

    test, validate = train_test_split(validate, train_size=0.5,
                                        random_state=seed)
    
    return train, validate, test