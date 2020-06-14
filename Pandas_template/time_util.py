# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:18:40 2020

"""

import datetime
from datetime import datetime, timedelta

time_1 = "2020/06/14 11:51:11"
time_2 = "2019/12/27 12:56:19"

def df_time2std_time(time_1):

    time_1 = time_1.split(" ")

    t1_d = time_1[0].split("/")
    
    if (int(t1_d[1]) > 12) or (int(t1_d[2]) > 31):
        raise ValueError("Wrong Date Input") 
    
    t1_t = time_1[1].split(":")
    
    if (int(t1_t[0]) > 23) or (int(t1_t[1]) > 59) or (int(t1_t[2]) > 59):
        raise ValueError("Wrong Time Input")
    
    combined_list = t1_d + t1_t
        
    for i in range(len(combined_list[1:])):
        if (int(combined_list[i+1]) < 10) and (combined_list[i+1] != "0" + str(combined_list[i+1])):
            combined_list[i+1] = "0" + str(int(combined_list[i+1]))
            
    output = ""
    for i in combined_list:
        output += i
   
    return int(output)


def over_hourtime(time_1, time_2, num):
    """
    Returns bool. 
        Whether over timeor not 
        If over time -> True
        If not       -> False

    """
    time_1 = str(df_time2std_time(time_1))
    time_2 = str(df_time2std_time(time_2))
    
    
    t1 = datetime(int(time_1[:4]), int(time_1[4:6]), int(time_1[6:8]), int(time_1[8:10]), int(time_1[10:12]), int(time_1[12:14]))
    t2 = datetime(int(time_2[:4]), int(time_2[4:6]), int(time_2[6:8]), int(time_2[8:10]), int(time_2[10:12]), int(time_2[12:14]))
    
    if int(t2.strftime("%Y%m%d%H%M%S")) >= int(t1.strftime("%Y%m%d%H%M%S")):
        output = ((t2 - t1) >= timedelta(hours=num))
    elif int(t1.strftime("%Y%m%d%H%M%S")) >= int(t2.strftime("%Y%m%d%H%M%S")):
        output = ((t1 - t2) >= timedelta(hours=num))

    return output


def get_weekdays(time_1):
    """
    Returns int.
        Sunday   -> 0
        Monday   -> 1
        Saturday -> 6
    """
    time_1 = str(df_time2std_time(time_1))
    output = datetime(int(time_1[:4]), int(time_1[4:6]), int(time_1[6:8])).strftime("%w")

    return output
    
def get_seasons(time_1):
    """
    Returns int.
        Winter -> 0
        Spring -> 1
        Summer -> 2
        Autumn -> 3
    """
    
    time_1 = str(df_time2std_time(time_1))
    season_list = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]

    month = int(time_1[4:6])
    print(month)
    
    if month in season_list[0]:
        season = 0
    elif month in season_list[1]:
        season = 1
    elif month in season_list[2]:
        season = 2
    elif month in season_list[3]:
        season = 3
    
    return season



if __name__ == "__main__":
    std_time = df_time2std_time(time_1)
    ot = over_hourtime(time_1, time_2, 1)
    week_days = get_weekdays(time_1)
    season = get_seasons(time_1)

