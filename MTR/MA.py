import numpy as np
import pandas as pd

def MA(s):
        date = ['1-8', '1-9', '1-10', '1-11', '1-12', '1-15', '1-16', '1-17', '1-18', '1-19', '1-22', '1-23', '1-24', '1-25',
                '1-26', '1-29', '1-30', '1-31', '2-1', '2-2', '2-5', '2-6', '2-7', '3-1', '3-2', '3-5', '3-6', '3-7', '3-8',
                '3-9', '3-12', '3-13', '3-14', '3-15', '3-16', '3-19', '3-20', '3-21', '3-22', '3-23', '3-26', '3-27', '3-28',
                '3-29', '3-30', '4-2', '4-3', '4-4', '4-9', '4-10', '4-11', '4-12', '4-13', '4-16', '4-17', '4-18', '4-19',
                '4-20', '4-23', '4-24', '4-25', '4-26', '4-27']
        s.columns = date
        # Disruption and public holidays
        incident_date = ['1-11', '1-12', '1-15', "1-23", '3-15', '3-16', '3-21', '3-22', '3-23', '3-30', "4-2", "4-3", "4-4"]
        normal_date = [i for i in date if i not in incident_date]
        normal_m = s[normal_date]
        normal_MA = normal_m.rolling(window=5, axis=1).mean()
        normal_MA.iloc[:,:5] = normal_MA[['1-22','1-30', '1-24', '1-25', '1-31']].values
        dev_MA = normal_m-normal_MA
        normal_sel = ['1-8', '1-9', '1-10', '1-16', '1-17', '1-8', '1-9', '1-10', '1-16', '1-17', '1-8', '1-9', '1-10']
        dev_MA[incident_date] = s[incident_date].values-normal_MA[normal_sel].values
        dev_MA = dev_MA[date]
        return normal_MA, dev_MA
