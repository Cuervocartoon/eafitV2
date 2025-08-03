import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json
import os

# --- CONFIGURACIÓN GLOBAL ---

def setup_environment():
    """
    Carga los datos, el tema de la marca y crea el directorio de salida.
    Retorna el DataFrame y la configuración de la marca.
    """
    # Crear directorio para guardar los gráficos si no existe
    if not os.path.exists('charts'):
        os.makedirs('charts')
        print("Directorio 'charts/' creado.")

    try:
        # Cargar el dataset de empresas
        df = pd.read_csv('empresasEafit.csv')
        
        # Cargar la identidad de marca desde el JSON
        with open('eafitBrand.json', 'r', encoding='utf-8') as f:
            brand_config = json.load(f)
        
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que los archivos 'empresasEafit.csv' y 'eafitBrand.json' estén en el mismo directorio.")
        return None, None

    # --- Creación del Tema de Plotly ---
    chart_colors = brand_config['colorPalette']['chartColors']
    eafit_template = go.layout.Template()
    eafit_template.layout.font = {
        'family': brand_config['typography']['defaultFontFamily'],
        'size': 12,
        'color': brand_config['colorPalette']['text']['primary']
    }
    eafit_template.layout.paper_bgcolor = brand_config['colorPalette']['background']['main']
    eafit_template.layout.plot_bgcolor = brand_config['colorPalette']['background']['highlight']
    eafit_template.layout.colorway = chart_colors
    eafit_template.layout.title = {
        'font': {
            'family': brand_config['typography']['styles']['heading2']['fontFamily'],
            'size': 18,
            'color': brand_config['colorPalette']['text']['primary']
        },
        'x': 0.5 # Centrar título
    }
    
    # Registrar y activar el tema
    pio.templates['eafit_brand'] = eafit_template
    pio.templates.default = 'eafit_brand'
    
    print("Entorno configurado exitosamente.")
    return df, brand_config

def save_chart_as_html(fig, filename):
    """
    Guarda una figura de Plotly como un archivo HTML autocontenido en la carpeta 'charts'.
    """
    # Generar el HTML del gráfico
    html_div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    
    # Crear el contenido del archivo HTML completo
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; }}
            .js-plotly-plot .plotly .modebar {{ right: 5px !important; top: 5px !important; }}
        </style>
    </head>
    <body>
        {html_div}
    </body>
    </html>
    """
    
    # Guardar el archivo
    filepath = os.path.join('charts', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"Gráfico guardado en: {filepath}")

# --- FUNCIONES DE GENERACIÓN DE GRÁFICOS ---

def generate_chart_01_radar_macroeconomic(df):
    """1. Radar de Desempeño por Macrosector."""
    radar_data = df.groupby(['Macrosector', 'Nombre Pilar'])['valoracionPonderada'].mean().reset_index()
    fig = px.line_polar(radar_data,
                        r='valoracionPonderada',
                        theta='Nombre Pilar',
                        color='Macrosector',
                        line_close=True,
                        markers=True,
                        title='Desempeño Promedio por Pilar y Macrosector')
    fig.update_layout(
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.4, 'xanchor': 'center', 'x': 0.5},
        margin=dict(b=150)
    )
    save_chart_as_html(fig, '01_radar_macroeconomic.html')

def generate_chart_02_bar_performance_by_pillar(df):
    """2. Desempeño Promedio General por Pilar."""
    data = df.groupby('Nombre Pilar')['valoracionPonderada'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(data, 
                 x='Nombre Pilar', 
                 y='valoracionPonderada', 
                 title='Desempeño Promedio por Pilar de Sostenibilidad',
                 labels={'valoracionPonderada': 'Valoración Ponderada Media (%)', 'Nombre Pilar': 'Pilar de Sostenibilidad'},
                 text_auto='.2f')
    fig.update_traces(textangle=0, textposition="outside")
    save_chart_as_html(fig, '02_bar_performance_by_pillar.html')

def generate_chart_03_treemap_companies_by_sector(df):
    """3. Distribución de Empresas por Macrosector."""
    data = df.groupby('Macrosector')['Razón social'].nunique().reset_index()
    fig = px.treemap(data, 
                     path=[px.Constant("Todos los Sectores"), 'Macrosector'], 
                     values='Razón social',
                     title='Distribución de Empresas Analizadas por Macrosector')
    fig.update_traces(textinfo="label+value+percent root")
    save_chart_as_html(fig, '03_treemap_companies_by_sector.html')

def generate_chart_04_box_performance_distribution(df):
    """4. Distribución del Desempeño por Pilar."""
    fig = px.box(df, 
                 x='Nombre Pilar', 
                 y='valoracionPonderada',
                 title='Distribución de la Valoración por Pilar',
                 labels={'valoracionPonderada': 'Valoración Ponderada (%)', 'Nombre Pilar': 'Pilar de Sostenibilidad'},
                 points="all")
    fig.update_xaxes(tickangle=45)
    save_chart_as_html(fig, '04_box_performance_distribution.html')

def generate_chart_05_scatter_income_vs_performance(df):
    """5. Relación entre Ingresos y Desempeño en Sostenibilidad."""
    df_chart = df.copy()
    
    # CORRECCIÓN: Convertir la columna de ingresos a tipo numérico.
    # 'errors='coerce'' convierte los valores no numéricos en NaN (Not a Number).
    df_chart['Ingresos operacionales'] = pd.to_numeric(df_chart['Ingresos operacionales'], errors='coerce')
    # Rellenar los valores NaN con 0 para poder agrupar.
    df_chart['Ingresos operacionales'].fillna(0, inplace=True)

    company_performance = df_chart.groupby('Razón social').agg(
        total_performance=('valoracionPonderada', 'sum'),
        income=('Ingresos operacionales', 'first'),
        macrosector=('Macrosector', 'first')
    ).reset_index()
    
    # Filtrar datos con ingresos > 0 para poder usar la escala logarítmica.
    plot_data = company_performance[company_performance['income'] > 0]
    
    fig = px.scatter(plot_data, 
                     x='income', 
                     y='total_performance',
                     size='income',
                     color='macrosector',
                     hover_name='Razón social',
                     title='Ingresos Operacionales vs. Desempeño Total en Sostenibilidad',
                     labels={'income': 'Ingresos Operacionales (escala log)', 'total_performance': 'Suma de Valoración Ponderada', 'macrosector': 'Macrosector'},
                     log_x=True)
    save_chart_as_html(fig, '05_scatter_income_vs_performance.html')

def generate_chart_06_bar_multinational_comparison(df):
    """6. Comparativa de Desempeño: Multinacionales vs. Nacionales."""
    data = df.groupby('¿Multinacional? Si/No')['valoracionPonderada'].mean().reset_index()
    fig = px.bar(data, 
                 x='¿Multinacional? Si/No', 
                 y='valoracionPonderada',
                 color='¿Multinacional? Si/No',
                 title='Desempeño Promedio: Multinacional vs. Nacional',
                 labels={'valoracionPonderada': 'Valoración Ponderada Media (%)', '¿Multinacional? Si/No': 'Tipo de Empresa'},
                 text_auto='.2f')
    save_chart_as_html(fig, '06_bar_multinational_comparison.html')

def generate_chart_07_bar_listed_comparison(df):
    """7. Comparativa: Empresas que cotizan en bolsa vs. las que no."""
    data = df.groupby('¿Cotiza en bolsa? Si/No')['valoracionPonderada'].mean().reset_index()
    fig = px.bar(data, 
                 x='¿Cotiza en bolsa? Si/No', 
                 y='valoracionPonderada',
                 color='¿Cotiza en bolsa? Si/No',
                 title='Desempeño: Cotiza en Bolsa vs. No Cotiza',
                 labels={'valoracionPonderada': 'Valoración Ponderada Media (%)', '¿Cotiza en bolsa? Si/No': '¿Cotiza en Bolsa?'},
                 text_auto='.2f')
    save_chart_as_html(fig, '07_bar_listed_comparison.html')

def generate_chart_08_sunburst_blocks_and_pillars(df):
    """8. Desglose Jerárquico por Bloque y Pilar."""
    data = df.groupby(['Bloque', 'Nombre Pilar'])['valoracionPonderada'].mean().reset_index()
    fig = px.sunburst(data, 
                      path=['Bloque', 'Nombre Pilar'], 
                      values='valoracionPonderada',
                      title='Desempeño Jerárquico: Bloques y Pilares')
    save_chart_as_html(fig, '08_sunburst_blocks_and_pillars.html')

def generate_chart_09_bar_top10_companies(df):
    """9. Top 10 Empresas por Desempeño Total."""
    company_performance = df.groupby('Razón social')['valoracionPonderada'].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(company_performance, 
                 y='Razón social', 
                 x='valoracionPonderada',
                 orientation='h',
                 title='Top 10 Empresas por Desempeño en Sostenibilidad',
                 labels={'valoracionPonderada': 'Suma de Valoración Ponderada', 'Razón social': 'Empresa'},
                 text='valoracionPonderada')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    save_chart_as_html(fig, '09_bar_top10_companies.html')

def generate_chart_10_bar_bottom10_companies(df):
    """10. Últimas 10 Empresas por Desempeño Total."""
    company_performance = df.groupby('Razón social')['valoracionPonderada'].sum().sort_values(ascending=True).head(10).reset_index()
    fig = px.bar(company_performance, 
                 y='Razón social', 
                 x='valoracionPonderada',
                 orientation='h',
                 title='Últimas 10 Empresas por Desempeño en Sostenibilidad',
                 labels={'valoracionPonderada': 'Suma de Valoración Ponderada', 'Razón social': 'Empresa'},
                 text='valoracionPonderada')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total descending'})
    save_chart_as_html(fig, '10_bar_bottom10_companies.html')
    
def generate_chart_11_histogram_foundation_year(df):
    """11. Distribución de Empresas por Año de Fundación."""
    df_chart = df.copy()
    
    # CORRECCIÓN: Asegurar que el año de fundación sea numérico.
    df_chart['Año de fundación'] = pd.to_numeric(df_chart['Año de fundación'], errors='coerce')
    df_chart.dropna(subset=['Año de fundación'], inplace=True) # Eliminar filas sin año válido
    
    unique_companies = df_chart.drop_duplicates(subset='Razón social')
    fig = px.histogram(unique_companies, 
                       x='Año de fundación',
                       title='Distribución de Empresas por Antigüedad (Año de Fundación)',
                       labels={'Año de fundación': 'Año de Fundación'},
                       nbins=20)
    save_chart_as_html(fig, '11_histogram_foundation_year.html')

def generate_chart_12_pie_property_type(df):
    """12. Proporción de Empresas por Tipo de Propiedad."""
    data = df.groupby('Tipo de propiedad (Privada, Pública, Mixta)')['Razón social'].nunique().reset_index()
    fig = px.pie(data, 
                 names='Tipo de propiedad (Privada, Pública, Mixta)', 
                 values='Razón social',
                 title='Proporción de Empresas por Tipo de Propiedad',
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    save_chart_as_html(fig, '12_pie_property_type.html')

def generate_chart_13_heatmap_materiality_vs_performance(df):
    """13. Mapa de Calor: Materialidad vs. Desempeño por Pilar."""
    df_chart = df.copy()
    
    # CORRECCIÓN: Limpiar y convertir columnas a tipo numérico.
    df_chart['Ponderación Materialidad'] = df_chart['Ponderación Materialidad'].astype(str).str.replace('%', '', regex=False)
    df_chart['Ponderación Materialidad'] = pd.to_numeric(df_chart['Ponderación Materialidad'], errors='coerce').fillna(0)
    df_chart['Valoración'] = pd.to_numeric(df_chart['Valoración'], errors='coerce').fillna(0)

    heatmap_data = df_chart.groupby('Nombre Pilar').agg(
        mean_performance=('Valoración', 'mean'),
        mean_materiality=('Ponderación Materialidad', 'mean')
    ).reset_index()
    
    fig = px.scatter(heatmap_data,
                     x='mean_materiality',
                     y='mean_performance',
                     text='Nombre Pilar',
                     size='mean_performance',
                     color='mean_materiality',
                     color_continuous_scale=px.colors.sequential.Cividis_r,
                     title='Análisis de Materialidad: Importancia vs. Desempeño',
                     labels={
                         'mean_materiality': 'Importancia Media de Materialidad (%)',
                         'mean_performance': 'Desempeño Medio en Valoración (%)'
                     })
    fig.update_traces(textposition='top center')
    save_chart_as_html(fig, '13_scatter_materiality_vs_performance.html')

def generate_chart_14_bar_performance_by_macrosector(df):
    """14. Desempeño Promedio por Macrosector."""
    data = df.groupby('Macrosector')['valoracionPonderada'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(data, 
                 x='Macrosector', 
                 y='valoracionPonderada', 
                 title='Ranking de Desempeño Promedio por Macrosector',
                 labels={'valoracionPonderada': 'Valoración Ponderada Media (%)', 'Macrosector': 'Macrosector'},
                 text_auto='.2f')
    fig.update_traces(textangle=0, textposition="outside")
    save_chart_as_html(fig, '14_bar_performance_by_macrosector.html')

def generate_chart_15_bar_family_business_comparison(df):
    """15. Comparativa: Empresas Familiares vs. No Familiares."""
    data = df.groupby('¿Es empresa familiar? Si/No')['valoracionPonderada'].mean().reset_index()
    fig = px.bar(data, 
                 x='¿Es empresa familiar? Si/No', 
                 y='valoracionPonderada',
                 color='¿Es empresa familiar? Si/No',
                 title='Desempeño Promedio: Empresa Familiar vs. No Familiar',
                 labels={'valoracionPonderada': 'Valoración Ponderada Media (%)', '¿Es empresa familiar? Si/No': '¿Es Empresa Familiar?'},
                 text_auto='.2f')
    save_chart_as_html(fig, '15_bar_family_business_comparison.html')


# --- EJECUCIÓN PRINCIPAL ---

def main():
    """
    Función principal que ejecuta la generación de todos los gráficos.
    """
    df, brand_config = setup_environment()
    
    if df is not None and brand_config is not None:
        print("\n--- Iniciando generación de gráficos ---")
        generate_chart_01_radar_macroeconomic(df)
        generate_chart_02_bar_performance_by_pillar(df)
        generate_chart_03_treemap_companies_by_sector(df)
        generate_chart_04_box_performance_distribution(df)
        generate_chart_05_scatter_income_vs_performance(df)
        generate_chart_06_bar_multinational_comparison(df)
        generate_chart_07_bar_listed_comparison(df)
        generate_chart_08_sunburst_blocks_and_pillars(df)
        generate_chart_09_bar_top10_companies(df)
        generate_chart_10_bar_bottom10_companies(df)
        generate_chart_11_histogram_foundation_year(df)
        generate_chart_12_pie_property_type(df)
        generate_chart_13_heatmap_materiality_vs_performance(df)
        generate_chart_14_bar_performance_by_macrosector(df)
        generate_chart_15_bar_family_business_comparison(df)
        print("\n--- Proceso de generación de gráficos completado. ---")
        print(f"Se han creado 15 archivos .html en la carpeta '{os.path.join(os.getcwd(), 'charts')}'")

if __name__ == '__main__':
    main()
