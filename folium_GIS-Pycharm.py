import pandas as pd
import folium
import os
import xlrd
from pandas import ExcelWriter
from pandas import ExcelFile

from IPython.display import display
import geopandas as gpd


#-----------
state_geo = ('IndianState.json')    # json data for Indian states: https://gist.github.com/ProProgrammer/781d5fbcb1d4364616c5
state_data = pd.read_excel('00-Ind.xlsx',sheet_name="states")

#-----------------------------------------------------------------------------

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5) # indian map coordinates taken from google

folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['States','Total'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Annual Estimate of birth rate India'
).add_to(m)

folium.LayerControl().add_to(m)

m.save('folium_pycharm.html')

display(m)
#-----------------------------------------------------------------------------------------