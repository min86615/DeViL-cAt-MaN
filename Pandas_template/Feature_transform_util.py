# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:35:43 2020

@author: Username
"""
import pandas as pd
import datetime
from datetime import datetime, date
 
### Get weekdays

day = "16/5/2019"

day_info = day.split("/")
anyday = datetime(int(day_info[2]), int(day_info[1]), int(day_info[0])).strftime("%w")
print(anyday)

def get_weekdays(day):
    """
    Input 16/5/2019 -> 2019-5-16
    
    Returns weekdays

    """
    
    day_info = day.split("/")
    anyday = datetime(int(day_info[2]), int(day_info[1]), int(day_info[0])).strftime("%w")
    # print(anyday)
    
    return anyday



### Get season

season_list = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]

month = int(day_info[1])

if month in season_list[0]:
    season = 0
elif month in season_list[1]:
    season = 1
elif month in season_list[2]:
    season = 2
elif month in season_list[3]:
    season = 3
else:
    season = 0

print(season)

def get_seasons(day):
    """
    Input 16/5/2019 -> 2019-5-16
    
    Returns seasons

    """
    
    day_info = day.split("/")
    season_list = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]

    month = int(day_info[1])
    
    if month in season_list[0]:
        season = 0
    elif month in season_list[1]:
        season = 1
    elif month in season_list[2]:
        season = 2
    elif month in season_list[3]:
        season = 3
    else:
        season = 0
    
    return season


df2 = pd.DataFrame(
        [["Green", "M", 10.1, 1],
         ["Red", "L", 9.4, 2], 
         ["Blue", "XL", 13.1, 1]]      
        )
df2.columns = ["Color", "Size", "Price", "Label"]

def abc(x):
    if float(x) > 10:
        output = 10
    else:
        output = 0
    return output

df2["Price"] = df2["Price"].apply(abc)




        
        