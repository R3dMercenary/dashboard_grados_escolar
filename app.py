#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import geopandas as gpd
import folium
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import warnings

# Silenciar SettingWithCopyWarning
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Silenciar FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Silenciar UserWarning de geometrías
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS") 

#######################
# Page configuration
st.set_page_config(
    page_title="Educación y Salud Mental",
    page_icon="𝚿",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


#######################

shapefile_path = "data/2023_1_00_ENT.shp"  # Replace with your shapefile path
gdf = gpd.read_file(shapefile_path)

# Ensure the GeoDataFrame has latitude and longitude columns for centroids
gdf['lon'] = gdf.geometry.centroid.x
gdf['lat'] = gdf.geometry.centroid.y
gdf['CVE_ENT']=gdf['CVE_ENT'].astype('int')

# Initialize session state to store the selected CVE_ENT
if "selected_cve_ent" not in st.session_state:
    st.session_state.selected_cve_ent = None

# Load data
@st.cache_data
def load_data(data_file):
    df=pd.read_parquet(f"data/{data_file}.parquet",engine='pyarrow')
    return df


population=load_data('population')
nivel_nombre_order = [
    'Ninguno',
    'Preescolar',
    'Primaria',
    'Secundaria', 
    'Preparatoria o bachillerato',
    'Profesional',
    'Normal',
    'Carrera técnica o comercial',
    'Maestría',
    'Doctorado'
]

# Convert 'nivel_nombre' column to a categorical type with the defined order
population['nivel_nombre'] = pd.Categorical(population['nivel_nombre'], categories=nivel_nombre_order, ordered=True)


income=load_data('income')
df=pd.merge(income,population, on=['folioviv','foliohog','numren','year','entidad'], how='inner')


background_color = "#1E1E1E"  # Dark gray
text_color = "#FFFFFF"  # White


####################### Sidebar
with st.sidebar:
    st.title('📏 Análisis de Grado Escolar por Entidad ')
    
    year_list = list(df.year.unique())[::-1]
    entidades=list(population.nombre_entidad.unique())[::-1]
    entidades.append('Nacional')

    tables=[
     'Distribución de niveles educativos aprobados por generación'    
    ,'Distribución del ingreso mensual promedio según nivel educativo aprobado por año'
    ,'Evolución de proporciones de niveles educativos aprobados por año'
    ,'Tasa de alfabetismo por año']

    
    selected_entidad = st.selectbox('Selecciona una entidad', entidades,entidades.index('Sonora'))
    selected_year = st.selectbox('Select a year', year_list)
    selected_table= st.selectbox('Tablas ', tables)
    
    df_selected_year = df[df.year == selected_year]

    
        
    

#######################
# Plots
    

def plot_literacy_proportion_over_time(population, selected_entidad,text_color,background_color):
    # Filter the data for the specific 'entidad'
    df = population[['folioviv', 'foliohog', 'numren', 'alfabetism', 'year', 'entidad', 'sexo', 'nombre_entidad']]
    is_entidad = df['nombre_entidad'] == selected_entidad
    if selected_entidad!='Nacional':
        df = df[is_entidad]
        
    
    df = df.groupby('year', as_index=False)['alfabetism'].value_counts(normalize=True)
    
    
    # Filter for both alfabetism levels: 1 and 2
    df_alfabetism_1 = df[df['alfabetism'] == 1]
    

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for alfabetism == 1
    ax.plot(
        df_alfabetism_1['year'],
        df_alfabetism_1['proportion'],
        marker='o',
        label='Alfabetas',
        color='white'
    )

    
    fig.patch.set_facecolor(background_color)  # Figure background
    ax.set_facecolor(background_color)

    # Customize the plot
    ax.set_title(f"Tasa de alfabetismo por año ({selected_entidad})",color=text_color, fontsize=14)
    ax.set_xlabel("Year", fontsize=12,color=text_color)
    ax.set_ylabel("Proportion", fontsize=12,color=text_color)
    ax.set_xticks(df['year'].unique())
    ax.tick_params(colors=text_color)
    ax.legend(title="Alfabetism Level")
    ax.grid(True, linestyle='--', alpha=0.6)

    

    # Show the plot in the Streamlit app
    plt.tight_layout()
    st.pyplot(fig)


def plot_proportion_by_year(population,selected_entidad,text_color,background_color):
    # Select necessary columns
    df = population[['folioviv', 'foliohog', 'numren', 'nivel_nombre', 'year', 'entidad', 'sexo', 'nombre_entidad']]
    is_entidad = df['nombre_entidad'] == selected_entidad
    if selected_entidad!='Nacional':
        df = df[is_entidad]

    # Grouping and calculating proportions
    df_counts = (
        df.groupby(['year'])['nivel_nombre']
        .value_counts(normalize=True)
        .rename('percentage')
        .reset_index()
    )

    # Pivoting the data to get a better structure for plotting
    df_pivot = df_counts.pivot(index='year', columns='nivel_nombre', values='percentage').fillna(0)

    # Plotting the bar graph
    fig, ax = plt.subplots(figsize=(10, 6))
    df_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

    
    fig.patch.set_facecolor(background_color)  # Figure background
    ax.set_facecolor(background_color)

    # Customizing the plot
    ax.set_title(f'Proporciones de Grado Aprobado por Año ( {selected_entidad} )',color=text_color)
    ax.set_xlabel('Año',color=text_color)
    ax.set_ylabel('Porcentaje',color=text_color)
    ax.tick_params(colors=text_color)
    ax.legend(title='Grado Escolar', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(fig)

def plot_proportion_by_generation(population, year,selected_entidad,text_color,background_color):
    # Grouping the data
    df=population
    is_entidad = df['nombre_entidad'] == selected_entidad
    if selected_entidad!='Nacional':
        df= df[is_entidad]

    data = df.groupby(['year', 'nivel_nombre'], as_index=False)['generacion'].value_counts(normalize=True)
    data = data[data['year'] == year]

    

    # Pivoting the data
    pivot_df = data.pivot(index='generacion', columns='nivel_nombre', values='proportion')

    # Plotting the data
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df.plot(kind='bar', ax=ax, colormap='viridis', width=0.8)

    
    fig.patch.set_facecolor(background_color)  # Figure background
    ax.set_facecolor(background_color)

    # Formatting the plot
    ax.set_title(f"Distribución de niveles educativos aprobados por generación ( {selected_entidad} {year} )",color=text_color)
    ax.set_ylabel("Taza ",color=text_color)
    ax.set_xlabel("Generación",color=text_color)
    ax.tick_params(colors=text_color)
    ax.legend(title="Generation", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Display the plot in the Streamlit dashboard
    st.pyplot(fig)

def plot_income_by_grade_and_year(income, population,selected_entidad,text_color,background_color):
    # Merging the dataframes
    df = pd.merge(income, population, on=['folioviv', 'foliohog', 'numren', 'year', 'entidad'], how='inner')

    # Selecting relevant columns
    df = df[['folioviv', 'foliohog', 'sexo', 'numren', 'edad', 'generacion',
             'entidad','nombre_entidad', 'ing_tri', 'nivel_nombre', 'year']]

    is_entidad = df['nombre_entidad'] == selected_entidad
    if selected_entidad!='Nacional':
        df = df[is_entidad]
                
    # Calculating total income per group
    df['ing_tri_total'] = df.groupby(['folioviv', 'foliohog', 'numren', 'year'])['ing_tri'].transform('sum')

    # Grouping by year and level name, calculating average income
    data = df.groupby(['year', 'nivel_nombre'], as_index=False)['ing_tri_total'].mean()
    
    # Calculating monthly income
    data['ing_mens'] = data['ing_tri_total'] / 3

    # Pivoting the data for the plot
    pivot_df = data.pivot(index='year', columns='nivel_nombre', values='ing_mens')

    # Plotting the bar graph
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df.plot(kind='bar', ax=ax, colormap='viridis', width=0.8)

    
    fig.patch.set_facecolor(background_color)  # Figure background
    ax.set_facecolor(background_color)

    # Formatting the plot
    ax.set_title(f"Evolución de proporciones de niveles educativos aprobados por año ( {selected_entidad} )",color=text_color)
    ax.set_ylabel("Ingreso Mensual (MXN)",color=text_color)
    ax.set_xlabel("Año",color=text_color)
    ax.tick_params(colors=text_color)
    ax.legend(title="Grado Escolar", bbox_to_anchor=(1.05, 1), loc='upper left')
    

    plt.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(fig)




# Heat map 


def heat_map(dataframe,gdf,year,text_color,background_color):
    df = dataframe

    nivel_nombre_order = [
        'Ninguno', 'Preescolar', 'Primaria', 'Secundaria',
        'Preparatoria o bachillerato', 'Profesional', 'Normal',
        'Carrera técnica o comercial', 'Maestría', 'Doctorado'
    ]

    # Grouping and processing the dataframe
    df_t = df.groupby(['entidad', 'year'])['nivelaprob'].mean().reset_index(name='nivel_promedio')

    df_t = df_t[df_t['year'] == year]
    


    # Merge the map data with the dataframe
    graph_df = pd.merge(df_t, gdf, left_on='entidad', right_on='CVE_ENT', how='inner')

    # Convert to GeoDataFrame for mapping
    data = gpd.GeoDataFrame(graph_df, geometry='geometry')

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(background_color)
    data.plot(
        column='nivel_promedio',
        cmap='viridis',       # Colormap for visualizing proportions
        linewidth=0.5,         # Line thickness for borders
        edgecolor='black',     # Border color between regions
        legend=True,           # Add a legend to show the color scale
        ax=ax                  # Plot on the provided axes
    )
    ax.set_axis_off()  # Remove axes for a clean map
    plt.title("Mapa de Grado Escolar aprovado Promedio",color=text_color)  
    legend = ax.get_legend()
    if legend:
        
        # Customize the legend text color
        for label in legend.get_texts():
            label.set_color(text_color)

    # Show the plot in Streamlit
    st.pyplot(fig)

# Display map
def display_map(gdf,selected_entidad):
    viridis_colormap = cm.get_cmap("viridis")
    dark_blue = [int(c * 255) for c in viridis_colormap(0)[:3]]  # Dark blue (start of viridis)
    yellow = [int(c * 255) for c in viridis_colormap(1)[:3]]
    
    gdf["fill_color"] = gdf['NOMGEO'].apply(
        lambda x: yellow if x == selected_entidad else dark_blue
    )

    deck = pdk.Deck(
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                data=gdf.__geo_interface__,  # Convert GeoDataFrame to GeoJSON
                get_fill_color="properties.fill_color",
                get_line_color="[200, 200, 200]",
                pickable=True,  # Enables interaction
                auto_highlight=True,
                highlight_color=yellow
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=gdf['lat'].mean(),
            longitude=gdf['lon'].mean(),
            zoom=4,
        ),
        tooltip={"html": "<b>Estado:</b> {NOMGEO}"},
    )

    # Render the map in Streamlit
    map_data = st.pydeck_chart(deck)
    return map_data



  
        




#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:


    with st.expander('About', expanded=True):
        st.write('''
            - Fuente de Datos: [ENIGH](https://www.inegi.org.mx/programas/enigh/nc/2022/).
            ''')
    st.markdown('''
    - :orange[**Grado Escolar**]: El promedio escolar aprobado a nivel nacional es la Secundaria. En general parece ser que hay una tendencia de aumento de nivel escolar con las nuevas generaciones.
    - :orange[**Ingresos**]: Se confirma que en promedio las personas con mayor grado escolar aprobado tienen mejor ingreso mensual.
            

    ''')

with col[1]:
    st.markdown(f'#### {selected_table}')
    st.markdown(' Tablas de análisis con el fin de contrastar resultados nacionales con resultados estatales')
    col1, col2 = st.columns(2)

    # Nacional
    with col1:
        st.markdown('##### Nacional')

        if selected_table=='Distribución de niveles educativos aprobados por generación':
            plot_proportion_by_generation(population, selected_year,'Nacional',text_color,background_color)
        
        if selected_table=='Distribución del ingreso mensual promedio según nivel educativo aprobado por año':
            plot_income_by_grade_and_year(income, population,'Nacional',text_color,background_color)

        if selected_table=='Evolución de proporciones de niveles educativos aprobados por año':
            plot_proportion_by_year(population,'Nacional',text_color,background_color)
        
        if selected_table=='Tasa de alfabetismo por año':
            plot_literacy_proportion_over_time(population,'Nacional',text_color,background_color)



    # Estatal
    with col2:
    
        st.markdown('##### Estatal')
        
        if selected_table=='Distribución de niveles educativos aprobados por generación':
            plot_proportion_by_generation(population, selected_year,selected_entidad,text_color,background_color)

        if selected_table=='Distribución del ingreso mensual promedio según nivel educativo aprobado por año':
            plot_income_by_grade_and_year(income, population,selected_entidad,text_color,background_color)

        if selected_table=='Evolución de proporciones de niveles educativos aprobados por año':
            plot_proportion_by_year(population,selected_entidad,text_color,background_color)

        if selected_table=='Tasa de alfabetismo por año':
            plot_literacy_proportion_over_time(population,selected_entidad,text_color,background_color)
        
    st.markdown('#### Grado Escolar Aprobado Promedio')
    heat_map(population,gdf,selected_year,text_color,background_color)
    #map_data = display_map(gdf,selected_entidad)

with col[2]:
    st.markdown('#### Generaciones vs Ingreso Mensual Promedio')
    df_selected_year['ing_tri_total'] = df_selected_year.groupby(['folioviv', 'foliohog', 'numren', 'year'])['ing_tri'].transform('sum')
    df_selected_year_sorted=df_selected_year.groupby(['year','generacion'],as_index=False)['ing_tri_total'].mean().sort_values(by="ing_tri_total", ascending=False)
    df_selected_year_sorted['ing_tri_total']=df_selected_year_sorted['ing_tri_total']/3
    df_selected_year_sorted['ing_tri_total']=df_selected_year_sorted['ing_tri_total'].round()

    st.dataframe(df_selected_year_sorted,
                 column_order=("generacion", "ing_tri_total"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "generacion": st.column_config.TextColumn(
                        "Generacion",
                    ),
                    "ing_tri_total": st.column_config.ProgressColumn(
                        "Ingreso Mensual",
                        format="%f",
                        min_value=0,
                        max_value=max(df_selected_year_sorted.ing_tri_total),
                        
                     )}
                 )

    df_grouped = df_selected_year.groupby('nivelaprob', as_index=False).first()[['nivelaprob', 'nivel_nombre']]
    df_grouped=df_grouped.sort_values('nivelaprob',ascending=False)
    st.markdown('#### Claves de Grados Escolares')
    st.dataframe(df_grouped,
                 column_order=("nivelaprob", "nivel_nombre"),
                 hide_index=True,
                 width=None,
                 column_config={
                    "nivelaprob": st.column_config.TextColumn(
                        "Clave",
                    ),
                    "nivel_nombre": st.column_config.TextColumn(
                        "Grado Escolar",
        
                        
                     )}
                 )
    
    