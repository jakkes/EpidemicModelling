import pandas
import numpy as np

"""
Reads Swedish population data from file.

Parameters
----------
filename: 
    The filename to parse.

age_groups: 
    List of 2-tuples defining the upper and lower limits (inclusive) of
    each age group. Use np.inf for unbounded intervals.

Returns
-------
output:
    ndarray with region as axis 0 and age groups along axis 1
"""
def parse_pop_swe(filename, age_groups):
    nages = len(age_groups)
    rawtable = pandas.read_csv(filename)
    rawtable[rawtable["ålder"] == "100+"] = "100"
    rawtable['ålder'] = rawtable['ålder'].astype(int)
    rawtable['2019'] = rawtable['2019'].astype(int)
    rawtable['region'] = rawtable['region'].str[3:-4] # remove region code
    ordered = rawtable.sort_values('region') # sort alphabetically by region name
    regions = pandas.unique(ordered['region']) # get a list of region names
    output = np.empty((regions.size, nages)) # create output array
    print(regions.size, nages, output.shape)
    for index, reg in enumerate(regions):
        #print(index)
        for age_index, age_interval in zip(range(nages), age_groups):
            lower, upper = age_interval
            n = rawtable[(rawtable['region'] == reg) &
                     (rawtable['ålder'] >= lower) &
                     (rawtable['ålder'] <= upper)].agg(np.sum)['2019']
            output[index, age_index] = n
            #print(output.shape)
    return output

if __name__ == "__main__":
    age_groups = [(0,9), (10,19), (20,29), (30,39), (40,49), (50,59),
                 (60,69),(70,79),(80,100)]
    filename = "~/git/EpidemicModeling/EpidemicModelling/data/sweden-regional/population2019.csv"
    data = parse_pop_swe(filename, age_groups)
    print(data.shape)