import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
np.random.seed(7)

# My env module
import src.env as env


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

##############################################################
  ##############       Primary Function       ##############
##############################################################

def get_zillow_big_query():

    if os.path.isfile('data/big_zillow_2017.csv'):
        df = pd.read_csv('data/big_zillow_2017.csv', index_col=0)

    else:
        df = big_query()

    return df



########################################
######  SQL Queries for Server  ########
########################################

def big_query():

    sql_query = '''
        SELECT prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        land.propertylandusedesc,
        story.storydesc,
        type.typeconstructiondesc
        FROM properties_2017 prop
        JOIN (
            SELECT parcelid, MAX(transactiondate) AS max_transactiondate
            FROM predictions_2017
            GROUP BY parcelid
            ) pred USING(parcelid)
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                        AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
        '''

    # reads the returned data tables into a dataframe
    df = pd.read_sql(sql_query, env.codeup_db('zillow'))

    # Cache data
    df.to_csv('data/big_zillow_2017.csv')
    
    return df