#!/usr/bin/env python
# coding: utf-8

# ### Best Buy Project
# ### Randy's Angels

# In[580]:


from datetime import datetime
t1 = datetime.now()
t1 = t1.strftime("%H:%M:%S")


# In[581]:


import pandas as pd
import numpy
import scipy
import os

df0 = pd.read_excel(os.path.expanduser('~/Downloads/Hackathon Data.xlsx'))


# In[582]:


df = df0.copy()
del df["PROMO_PRICE"], df["COMPETITOR_PRICE"]
df.head()


# In[584]:


def accuracy(y1,y2):
    import numpy as np
    from statsmodels.tools.eval_measures import rmse
    
    accuracy_df=pd.DataFrame()
    
    rms_error = np.round(rmse(y1, y2),1)
           
    accuracy_df=accuracy_df.append({"RMSE":rms_error}, ignore_index=True)
    
    return rms_error


# In[592]:


def pysnaive(train_series,seasonal_periods,forecast_horizon):
    '''
    Python implementation of Seasonal Naive Forecast. 
    This should work similar to https://otexts.com/fpp2/simple-methods.html
    Returns two arrays
     > fitted: Values fitted to the training dataset
     > fcast: seasonal naive forecast
    
    Author: Sandeep Pawar
    Edited by: A. T. Crawford
    
    Date: Apr 9, 2020
    Edited Date: Jan 18, 2023
    
    Ver: 1.1
    
    train_series is the training data you will use to train the model. 
    seasonal_periods is the 'lag' on the naive model. A value of '7' is a 1 week lag
    forecast_horizons is the number of days you want to predict for 
    '''
    import numpy as np
    
    if len(train_series)>= seasonal_periods: #checking if there are enough observations in the training data
        
        last_season=train_series.iloc[-seasonal_periods:]
        
        reps=np.int(np.ceil(forecast_horizon/seasonal_periods))
        
        fcarray=np.tile(last_season,reps)
        
        fcast=pd.Series(fcarray[:forecast_horizon])
        
        fitted = train_series.shift(seasonal_periods)
        
    else:
        fcast=print("Length of the trainining set must be greater than number of seasonal periods") 
    
    return fitted, fcast


# In[599]:


def all_sku(seasonal): 
    rmse_list = []
    sum_rmse = 0
    vals = df['Encoded_SKU_ID'].unique()
    vals = list(sorted(vals))
    for i in vals:
        #print(i)
        df2 = df.copy()
        df2 = df2[df2['Encoded_SKU_ID']==i]
        df2 = df2.set_index('SALES_DATE')
        df2 = df2.drop(columns = ['Encoded_SKU_ID' , 'RETAIL_PRICE', 'Inventory', 'class_code', 'subclass_code'], axis =1)
        training_set = df2[:-7]
        test_set = df2[-7:]
        liltrain = training_set
        
        py_snaive_fit = pysnaive(liltrain['DAILY_UNITS'], 
                     seasonal_periods=seasonal,
                     forecast_horizon=7)[0]

        #forecast
        py_snaive = pysnaive(liltrain['DAILY_UNITS'], 
                     seasonal_periods=seasonal,
                     forecast_horizon=7)[1]
        predictions["py_snaive"] = py_snaive.values 
        
        acc = accuracy(predictions['DAILY_UNITS'], predictions['py_snaive'])
        rmse_list.append(acc)
        sum_rmse+=acc
        
    return sum_rmse


# In[600]:


#min_rmse = 10000000
#for i in range(1,50):
#    sum_rmse = all_sku(i)
 #   #print(sum_rmse)
 #   #print(min_rmse)
 #   if sum_rmse < min_rmse:
 #       min_rmse = sum_rmse
  #      print(i)
    


# In[601]:


dfval = pd.read_excel(os.path.expanduser('~/Downloads/Validation_Data.xlsx'))
dfval = dfval.sort_index()


# In[602]:


dfval[dfval['Encoded_SKU_ID'] == 4]


# In[693]:


def all_sku_valid(seasonal): 
    rmse_list = []
    sum_rmse = 0
    vals = dfval['Encoded_SKU_ID'].unique() #we had to use only SKUs in the validation set since there were fewer than in the training set. 
    vals = list(sorted(vals))
    for i in vals:
        #print(i)
        dfvalid = dfval.copy()
        df2 = df.copy()
        df2 = df2[df2['Encoded_SKU_ID']==i]
        dfvalid = dfvalid[dfvalid['Encoded_SKU_ID']==i]
        df2 = df2.set_index('SALES_DATE')
        dfvalid = dfvalid.set_index('SALES_DATE')
        dfvalid = dfvalid.sort_index()
        df2 = df2.drop(columns = ['Encoded_SKU_ID' , 'RETAIL_PRICE', 'Inventory', 'class_code', 'subclass_code'], axis =1)
        dfvalid = dfvalid.drop(columns = ['Encoded_SKU_ID' , 'CLASS_NAME', 'SUBCLASS_NAME', 'ML_NAME', 'CATEGORY_NAME', 'RETAIL_PRICE', 'PROMO_PRICE', 'COMPETITOR_PRICE', 'Inventory', 
                                         'Forecasted Units', 'CP2','PP2'], axis =1)
        training_set = df2
        test_set = dfvalid
        liltrain = training_set[-60:] #last 60 entries of the original data
        predictions = dfvalid.copy() #predictions compared to the validation set 
        py_snaive_fit = pysnaive(liltrain['DAILY_UNITS'], 
                     seasonal_periods=seasonal,
                     forecast_horizon=7)[0]

        #forecast
        py_snaive = pysnaive(liltrain['DAILY_UNITS'], 
                     seasonal_periods=seasonal,
                     forecast_horizon=7)[1]
        predictions["py_snaive"] = py_snaive.values
        acc = accuracy(predictions['DAILY_UNITS'], predictions['py_snaive']) #
        rmse_list.append(acc)
        sum_rmse+=acc
        #print(i ,',', acc)
        
    return sum_rmse/len(vals)


# In[694]:


#This cell can be used to find the optimal lag by minimizing rmse.
#min_rmse = 10000000
#for i in range(1,50):
  #  sum_rmse = all_sku_valid(i)
    #print(sum_rmse)
    #print(min_rmse)
  #  if sum_rmse < min_rmse:
   #     min_rmse = sum_rmse
   #     print(i)


# In[695]:


all_sku_valid(7)


# ### Example Graph

# In[687]:


dfvalid = dfval.copy()
dfvalid = dfvalid[dfvalid['Encoded_SKU_ID']==557] #randomly generated number
dfvalid = dfvalid.set_index('SALES_DATE')
dfvalid = dfvalid.drop(columns = ['Encoded_SKU_ID' , 'CLASS_NAME', 'SUBCLASS_NAME', 'ML_NAME', 'CATEGORY_NAME', 'RETAIL_PRICE', 'PROMO_PRICE', 'COMPETITOR_PRICE', 'Inventory', 
                                         'Forecasted Units'], axis =1)


# In[688]:


dfvalid = dfvalid.sort_index()
#dfvalid


# In[689]:


trainer = df.copy()
trainer= trainer[trainer['Encoded_SKU_ID']==557]
trainer= trainer.drop(columns = ['Encoded_SKU_ID', 'RETAIL_PRICE', 'Inventory' ,'class_code', 'subclass_code'], axis=1)
trainer = trainer[-60:]
trainer = trainer.set_index('SALES_DATE')
#trainer


# In[692]:


preds = dfvalid.copy()
py_snaive_fit = pysnaive(trainer['DAILY_UNITS'], 
                     seasonal_periods=7,
                     forecast_horizon=7)[0]

#forecast
py_snaive = pysnaive(trainer['DAILY_UNITS'], 
                     seasonal_periods=7,
                     forecast_horizon=7)[1]
preds["py_snaive"] = py_snaive.values 


# In[691]:


pd.plotting.register_matplotlib_converters()
trainer['DAILY_UNITS'].plot(figsize=(12,8))#, style="--", color="gray", legend=True, label="Train")
py_snaive_fit.plot(color="g", legend=True, label="SNaive_Fitted")
preds["DAILY_UNITS"].plot(style="--",color="r", legend=True, label="Validation")
preds["py_snaive"].plot(color="b", legend=True, label="Snaive_fc");


# In[631]:


t2 = datetime.now()
t2 = t2.strftime("%H:%M:%S")


# In[614]:


t2


# In[615]:


t1


# Total run time was 2 minutes and 8 seconds

# In[ ]:




