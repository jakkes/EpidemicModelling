import matplotlib.pyplot as plt
import geopandas

sweden = geopandas.read_file("./data/sweden-regional/geo/Lan_Sweref99TM_region.shp")

sweden.plot()

plt.show()