import numpy as np
import pandas
import matplotlib.pyplot as plt
import descartes
import geopandas

def plot_sweden(data):
    """
    Data är pandas frame med kolumner "S", "E", "I", "R", "region".
    Kör python3 1d_model.py så får du in riktig data här
    """
    data = data.rename(columns={"S": "susceptible", "I": "infectious", "R": "recovered", "E": "exposed"})
    print(data.head())
    sweden = geopandas.read_file("./data/sweden-regional/geo/Lan_Sweref99TM_region.shp")
    #sweden = sweden.sort_values('LnNamn')
    #print(sweden.head())
   
    merged = sweden.set_index('LnNamn').join(data.set_index('region'))

    variable_to_plot = 'infectious'
    fig, ax = plt.subplots(1, figsize=(10,10))
    merged.plot(column=variable_to_plot, cmap='YlOrRd', ax=ax)
    ax.axis("off")
    ax.set_title(variable_to_plot.title())

    vmin, vmax = merged[variable_to_plot].agg([np.min, np.max])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))# empty array for the data rangesm._A = []# add the colorbar to the figurecbar = fig.colorbar(sm)
    cb = fig.colorbar(sm)
    cb.set_label(str.format("Number of {:s}", variable_to_plot))

    #sweden.plot()
    plt.show()
 #   pass

#if __name__ == "__main__":
 