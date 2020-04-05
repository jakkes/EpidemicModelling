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
    #print(data["S"].dtype)
    data["S"] = data["S"].astype(np.float64) # Datatyperna kommer in som object, måste castas till float för att geopandas ska funka
    data["E"] = data["E"].astype(np.float64)
    data["I"] = data["I"].astype(np.float64)
    data["R"] = data["R"].astype(np.float64)
    #print(data["S"].dtype)
    data = data.rename(columns={"S": "susceptible", "I": "infectious", "R": "recovered", "E": "exposed"})

    sweden = geopandas.read_file("./data/sweden-regional/geo/Lan_Sweref99TM_region.shp")

    merged = sweden.set_index('LnNamn').join(data.set_index('region'))

    variables_to_plot = ["susceptible", "infectious", "recovered"]
    colormaps = ['Blues', 'YlOrRd', 'YlGn']    
    fig, axs = plt.subplots(1, len(variables_to_plot), figsize=(10,10))
    for var, cm, ax in zip(variables_to_plot, colormaps, axs):
        merged.plot(column=var, cmap=cm, ax=ax)
        ax.axis("off")
        ax.set_title(var.title())

        vmin, vmax = merged[var].agg([np.min, np.max])
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label(str.format("Number of {:s}", var))

    plt.show()