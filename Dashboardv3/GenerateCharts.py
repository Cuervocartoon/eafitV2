import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Carga de la configuración de la marca EAFIT y creación de la plantilla de Plotly
# ---------------------------------------------------------------------------

# Cargar la configuración de la marca desde el archivo JSON
with open('eafitBrand.json', 'r') as f:
    eafit_brand = json.load(f)

# Extraer la paleta de colores y la configuración de tipografía
color_palette = eafit_brand['colorPalette']
typography = eafit_brand['typography']

# Crear una plantilla personalizada de Plotly
pio.templates['eafit_brand_template'] = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family=typography['defaultFontFamily'],
            size=12,
            color=color_palette['text']['primary']
        ),
        title_font=dict(
            family=typography['styles']['title']['fontFamily'],
            size=24,
            color=color_palette['text']['primary']
        ),
        paper_bgcolor=color_palette['background']['main'],
        plot_bgcolor=color_palette['background']['highlight'],
        colorway=color_palette['chartColors'],
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor=color_palette['background']['main']
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor=color_palette['background']['main']
        ),
        legend=dict(
            font=dict(size=12),
            title_font=dict(size=14)
        )
    )
)

# Establecer la plantilla personalizada como predeterminada
pio.templates.default = "eafit_brand_template"

# ---------------------------------------------------------------------------
# Carga y preprocesamiento de datos
# ---------------------------------------------------------------------------

raw_data = pd.read_csv('empresasEafit.csv')
raw_data.rename(columns=lambda x: x.strip(), inplace=True)
raw_data = raw_data[~raw_data['Macrosector'].isin(['0', 'No', 'No informa', 'SI', 'Si'])]

# ---------------------------------------------------------------------------
# Funciones de ayuda
# ---------------------------------------------------------------------------

def save_chart_as_html(fig, filename):
    """
    Guarda una figura de Plotly como un archivo HTML autocontenido en la carpeta 'charts'.
    """
    if not os.path.exists('charts'):
        os.makedirs('charts')
    
    filepath = os.path.join('charts', filename)
    pio.write_html(fig, file=filepath, auto_open=False, include_plotlyjs='cdn', full_html=True)

def wrap_text(text, max_length=25):
    """
    Envuelve el texto en varias líneas si excede la longitud máxima de caracteres.
    """
    if len(text) <= max_length:
        return text
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + len(current_line) > max_length and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '<br>'.join(lines)

# ---------------------------------------------------------------------------
# Generación de visualizaciones
# ---------------------------------------------------------------------------

# 1. Radar de Competitividad por Pilar y Macrosector
radar_final_df = (
    raw_data.groupby(['Macrosector', 'Nombre Pilar'])['valoracionPonderada']
    .mean()
    .reset_index()
    .pivot_table(
        index='Nombre Pilar',
        columns='Macrosector',
        values='valoracionPonderada'
    )
    .fillna(0)
    .reset_index()
    .rename(columns={'Nombre Pilar': 'category'})
)

categories = radar_final_df['category'].tolist()
macrosectores = radar_final_df.drop(columns='category').columns.tolist()
wrapped_categories = [wrap_text(cat) for cat in categories]

fig1 = go.Figure()

for i, macrosector in enumerate(macrosectores):
    values = radar_final_df[macrosector].values
    fig1.add_trace(go.Scatterpolar(
        r=values,
        theta=wrapped_categories,
        fill='toself',
        name=macrosector,
        opacity=0.6
    ))

fig1.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(radar_final_df.drop(columns='category').max()) * 1.1],
            tickmode='linear',
            tick0=0,
            dtick=5
        ),
        angularaxis=dict(
            tickmode='array',
            tickvals=list(range(len(categories))),
            ticktext=wrapped_categories,
            tickfont=dict(size=10),
            rotation=0
        )
    ),
    showlegend=True,
    title={
        'text': '1. Radar de Competitividad por Pilar y Macrosector',
        'x': 0.5,
        'xanchor': 'center'
    }
)
fig1.show()
save_chart_as_html(fig1, '01_radar_macroeconomic.html')
fig1.write_image("img/01_radar_macroeconomic.png", width=1000, height=800, scale=2)


# 2. Mapa de Calor de Desempeño Empresarial
heatmap_pivot = raw_data.pivot_table(
    index='Razón social',
    columns='Nombre Pilar',
    values='valoracionPonderada',
    aggfunc='sum',
    fill_value=0
)
heatmap_pivot.sort_values(by='Razón social', inplace=True)
wrapped_pilars = [wrap_text(col) for col in heatmap_pivot.columns]

fig2 = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=wrapped_pilars,
    y=heatmap_pivot.index,
    colorscale='Blues',
    colorbar=dict(title='Valoración Ponderada')
))

fig2.update_layout(
    title='2. Mapa de Calor de Desempeño Empresarial',
    xaxis_title='Pilares',
    yaxis_title='Empresas',
    width=1000,
    height=800
)
fig2.show()
save_chart_as_html(fig2, '02_heatmap_performance.html')
fig2.write_image("img/02_heatmap_performance.png", width=1000, height=800, scale=2)


# 3. Diagrama de Violín de la Dispersión del Desempeño Sectorial
total_scores = raw_data.groupby(['Razón social', 'Macrosector'])['valoracionPonderada'].sum().reset_index()
total_scores.rename(columns={'valoracionPonderada': 'puntaje_total'}, inplace=True)
total_scores.drop(columns='Razón social', inplace=True)

fig3 = px.violin(total_scores, y='puntaje_total', x='Macrosector', box=True, points='all',
               color='Macrosector', title='3. Violin Plot de Puntajes por Macrosector',
               labels={'puntaje_total': 'Puntaje Total', 'Macrosector': 'Macrosector'})
fig3.update_traces(meanline_visible=True)
fig3.update_layout(
    xaxis_title='Macrosector',
    yaxis_title='Puntaje Total'
)
fig3.show()
save_chart_as_html(fig3, '03_violin_plot_performance.html')
fig3.write_image("img/03_violin_plot_performance.png", width=1000, height=800, scale=2)


# 4. Treemap de Impacto: Sostenibilidad y Peso Económico
raw_data['valoracionPonderada'] = pd.to_numeric(raw_data['valoracionPonderada'], errors='coerce').fillna(0)
raw_data['Ingresos operacionales'] = pd.to_numeric(raw_data['Ingresos operacionales'], errors='coerce').fillna(0)
company_agg_data = raw_data.groupby('Razón social').agg(
    Puntaje_Total_Sostenibilidad=('valoracionPonderada', 'sum'),
    Ingresos_Operacionales=('Ingresos operacionales', 'first'),
    Macrosector=('Macrosector', 'first')
).reset_index()
company_agg_data = company_agg_data[company_agg_data['Ingresos_Operacionales'] > 0]

fig4 = px.treemap(
    company_agg_data,
    path=[px.Constant("Todas las Empresas"), 'Macrosector', 'Razón social'],
    values='Ingresos_Operacionales',
    color='Puntaje_Total_Sostenibilidad',
    hover_data={
        'Macrosector': True,
        'Ingresos_Operacionales': ':.2f',
        'Puntaje_Total_Sostenibilidad': ':.2f'
    },
    color_continuous_scale='Blues',
    color_continuous_midpoint=np.average(company_agg_data['Puntaje_Total_Sostenibilidad'], weights=company_agg_data['Ingresos_Operacionales'])
)
fig4.update_layout(
    title_text='<b>Treemap de Impacto: Sostenibilidad y Peso Económico</b><br><sup>El tamaño representa los Ingresos Operacionales, el color el Desempeño en Sostenibilidad</sup>',
    title_x=0.5,
    margin = dict(t=60, l=25, r=25, b=25)
)
fig4.update_traces(
    hovertemplate='<b>%{label}</b><br><br>' +
                  'Macrosector: %{customdata[0]}<br>' +
                  'Ingresos Operacionales (Tamaño): %{value:,.2f}<br>' +
                  'Puntaje Sostenibilidad (Color): %{color:.2f}<extra></extra>'
)
fig4.show()
save_chart_as_html(fig4, '04_treemap_sustainability.html')
fig4.write_image("img/04_treemap_sustainability.png", width=1000, height=800, scale=2)

# 5. Diagrama de Burbujas: Sostenibilidad vs. Antigüedad Empresarial
raw_data['Año de fundación'] = pd.to_numeric(raw_data['Año de fundación'], errors='coerce')
bubble_chart_data = raw_data.groupby('Razón social').agg(
    Puntaje_Total_Sostenibilidad=('valoracionPonderada', 'sum'),
    Ingresos_Operacionales=('Ingresos operacionales', 'first'),
    Macrosector=('Macrosector', 'first'),
    Ano_Fundacion=('Año de fundación', 'first')
).reset_index()
bubble_chart_data.dropna(subset=['Ano_Fundacion', 'Ingresos_Operacionales'], inplace=True)
bubble_chart_data = bubble_chart_data[bubble_chart_data['Ingresos_Operacionales'] > 0]
bubble_chart_data['Ano_Fundacion'] = bubble_chart_data['Ano_Fundacion'].astype(int)

fig5 = px.scatter(
    bubble_chart_data,
    x="Ano_Fundacion",
    y="Puntaje_Total_Sostenibilidad",
    size="Ingresos_Operacionales",
    color="Macrosector",
    hover_name="Razón social",
    size_max=60,
    log_x=False
)
fig5.update_layout(
    title='<b>Sostenibilidad vs. Antigüedad Empresarial</b><br><sup>El tamaño de la burbuja indica los ingresos operacionales</sup>',
    title_x=0.5,
    xaxis_title="Año de Fundación",
    yaxis_title="Puntaje Total de Sostenibilidad",
    legend_title_text='Macrosector'
)
fig5.update_traces(
    hovertemplate='<b>%{hovertext}</b><br><br>' +
                  'Año de Fundación: %{x}<br>' +
                  'Puntaje Sostenibilidad: %{y:.2f}<br>' +
                  'Ingresos Operacionales: %{marker.size:,.0f}<extra></extra>'
)
fig5.show()
save_chart_as_html(fig5, '05_bubble_chart_sustainability_vs_age.html')
fig5.write_image("img/05_bubble_chart_sustainability_vs_age.png", width=1000, height=800, scale=2)

# 6. Gráfico de Coordenadas Paralelas para Perfiles de Sostenibilidad
pivoted_data = raw_data.groupby(['Razón social', 'Nombre Pilar'])['valoracionPonderada'].sum().reset_index()
parallel_coords_df = pivoted_data.pivot(
    index='Razón social',
    columns='Nombre Pilar',
    values='valoracionPonderada'
).fillna(0).reset_index()
parallel_coords_df = pd.merge(
    parallel_coords_df,
    company_agg_data[['Razón social', 'Macrosector', 'Puntaje_Total_Sostenibilidad']],
    on='Razón social',
    how='left'
)
dimensions = list(parallel_coords_df.columns)
dimensions.remove('Razón social')
dimensions.remove('Macrosector')
dimensions.remove('Puntaje_Total_Sostenibilidad')

fig6 = px.parallel_coordinates(
    parallel_coords_df,
    color="Puntaje_Total_Sostenibilidad",
    dimensions=dimensions,
    labels={"Razón social": "Empresa", "Macrosector": "Sector"},
    color_continuous_scale=px.colors.sequential.Blues,
    title="Perfiles de Sostenibilidad por Empresa"
)
fig6.update_layout(
    title='<b>Perfiles de Sostenibilidad por Empresa</b><br><sup>Cada línea es una empresa, coloreada por su puntaje total</sup><br>',
    title_x=0.5,
    title_y=0.95,
)
fig6.show()
save_chart_as_html(fig6, '06_parallel_coordinates_sustainability_profiles.html')
fig6.write_image("img/06_parallel_coordinates_sustainability_profiles.png", width=1000, height=800, scale=2)

# 7. Diagrama Sankey de Flujo de Valoración
flow1 = raw_data.groupby(['Bloque', 'Nombre Pilar'])['valoracionPonderada'].sum().reset_index()
flow1.rename(columns={'Bloque': 'source', 'Nombre Pilar': 'target', 'valoracionPonderada': 'value'}, inplace=True)
flow2 = raw_data.groupby(['Nombre Pilar', 'Macrosector'])['valoracionPonderada'].sum().reset_index()
flow2.rename(columns={'Nombre Pilar': 'source', 'Macrosector': 'target', 'valoracionPonderada': 'value'}, inplace=True)
sankey_data = pd.concat([flow1, flow2], axis=0)
sankey_data = sankey_data[sankey_data['value'] > 0]
unique_nodes = pd.unique(sankey_data[['source', 'target']].values.ravel('K'))
node_mapping = {node: i for i, node in enumerate(unique_nodes)}
sankey_data['source_id'] = sankey_data['source'].map(node_mapping)
sankey_data['target_id'] = sankey_data['target'].map(node_mapping)

fig7 = go.Figure(data=[go.Sankey(
    node=dict(
      pad=15,
      thickness=20,
      line=dict(color="black", width=0.5),
      label=unique_nodes,
    ),
    link=dict(
      source=sankey_data['source_id'],
      target=sankey_data['target_id'],
      value=sankey_data['value']
  ))])
fig7.update_layout(
    title_text="<b>Diagrama Sankey del Flujo de Valoración de Sostenibilidad</b><br><sup>Flujo desde Bloque -> Pilar -> Macrosector</sup>",
    title_x=0.5
)
fig7.show()
save_chart_as_html(fig7, '07_sankey_diagram_sustainability_flows.html')
fig7.write_image("img/07_sankey_diagram_sustainability_flows.png", width=1000, height=800, scale=2)

# 8. Gráfico Solar (Sunburst) de la Jerarquía del Desempeño
sunburst_data = raw_data.dropna(subset=['Bloque', 'Nombre Pilar'])
sunburst_data = sunburst_data[sunburst_data['valoracionPonderada'] > 0]

fig8 = px.sunburst(
    sunburst_data,
    path=[px.Constant("Desempeño Total"), 'Bloque', 'Nombre Pilar'],
    values='valoracionPonderada',
    color='valoracionPonderada',
    color_continuous_scale='Blues',
    hover_data={'Bloque': True, 'Nombre Pilar': True}
)
fig8.update_layout(
    title_text="<b>Gráfico Solar de la Jerarquía del Desempeño en Sostenibilidad</b><br><sup>Tamaño y color representan la contribución de cada área</sup>",
    title_x=0.5,
    margin = dict(t=60, l=25, r=25, b=25)
)
fig8.update_traces(
    hovertemplate='<b>%{label}</b><br>Valoración Ponderada Total: %{value:,.2f}<br>Contribución al Padre: %{percentParent:.2%}<extra></extra>'
)
fig8.show()
save_chart_as_html(fig8, '08_sunburst_performance_hierarchy.html')
fig8.write_image("img/08_sunburst_performance_hierarchy.png", width=1000, height=800, scale=2)

# 9. Matriz de Correlación entre Pilares de Sostenibilidad
pillar_data = parallel_coords_df[dimensions]
correlation_matrix = pillar_data.corr()
fig9 = px.imshow(
    correlation_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='Blues', 
    zmin=-1, zmax=1
)
fig9.update_layout(
    title_text='<b>Matriz de Correlación entre Pilares de Sostenibilidad</b><br><sup>Revela sinergias (azul) y trade-offs (rojo)</sup>',
    title_x=0.5,
    xaxis_tickangle=-45
)
fig9.show()
save_chart_as_html(fig9, '09_correlation_matrix_sustainability_pillars.html')
fig9.write_image("img/09_correlation_matrix_sustainability_pillars.png", width=1000, height=800, scale=2)

# 10. Gráfico de Barras Divergentes: Desempeño Relativo al Sector
sector_avg_score = company_agg_data.groupby('Macrosector')['Puntaje_Total_Sostenibilidad'].mean().reset_index()
sector_avg_score.rename(columns={'Puntaje_Total_Sostenibilidad': 'Promedio_Sector'}, inplace=True)
diverging_data = pd.merge(company_agg_data, sector_avg_score, on='Macrosector')
diverging_data['Diferencia_vs_Promedio'] = diverging_data['Puntaje_Total_Sostenibilidad'] - diverging_data['Promedio_Sector']
diverging_data['Desempeño_Relativo'] = np.where(diverging_data['Diferencia_vs_Promedio'] >= 0, 'Superior al Promedio', 'Inferior al Promedio')
diverging_data.sort_values(by=['Macrosector', 'Diferencia_vs_Promedio'], inplace=True)

fig10 = px.bar(
    diverging_data,
    x='Diferencia_vs_Promedio',
    y='Razón social',
    color='Desempeño_Relativo',
    color_discrete_map={
        'Superior al Promedio': 'green',
        'Inferior al Promedio': 'red'
    },
    orientation='h',
    labels={'Diferencia_vs_Promedio': 'Diferencia vs. Promedio del Sector', 'Razón social': 'Empresa'}
)
fig10.update_layout(
    title='<b>Desempeño Relativo de Sostenibilidad vs. Promedio del Sector</b><br><sup>Barras verdes superan el promedio, rojas están por debajo</sup>',
    title_x=0.5,
    yaxis_title='Empresa',
    xaxis_title='Desviación del Promedio del Sector',
    height=max(600, len(diverging_data) * 20) 
)
fig10.show()
save_chart_as_html(fig10, '10_diverging_bar_performance_vs_sector.html')
fig10.write_image("img/10_diverging_bar_performance_vs_sector.png", width=1000, height=800, scale=2)

# 11. Gráfico de Cajas Comparativo: Propiedad y Desempeño (Pública vs. Privada)
property_type_data = raw_data[['Razón social', 'Tipo de propiedad (Privada, Pública, Mixta)']].drop_duplicates()
boxplot_data = pd.merge(company_agg_data, property_type_data, on='Razón social')
boxplot_data.dropna(subset=['Tipo de propiedad (Privada, Pública, Mixta)'], inplace=True)

fig11 = px.box(
    boxplot_data,
    x='Tipo de propiedad (Privada, Pública, Mixta)',
    y='Puntaje_Total_Sostenibilidad',
    color='Tipo de propiedad (Privada, Pública, Mixta)',
    notched=True,
    points="all"
)
fig11.update_layout(
    title='<b>Comparativa de Desempeño en Sostenibilidad por Tipo de Propiedad</b>',
    title_x=0.5,
    xaxis_title='Tipo de Propiedad de la Empresa',
    yaxis_title='Puntaje Total de Sostenibilidad',
    showlegend=False
)
fig11.show()
save_chart_as_html(fig11, '11_boxplot_performance_by_property_type.html')
fig11.write_image("img/11_boxplot_performance_by_property_type.png", width=1000, height=800, scale=2)

# 12. Gráfico de Densidad por Atributo: Multinacional vs. Nacional
multinational_info = raw_data[['Razón social', '¿Multinacional? Si/No']].drop_duplicates()
density_data = pd.merge(company_agg_data, multinational_info, on='Razón social')
density_data.dropna(subset=['¿Multinacional? Si/No'], inplace=True)
groups = {
    'Multinacional': 'Si',
    'Nacional': 'No'
}
colors = ['#1f77b4', '#ff7f0e']

fig12 = go.Figure()

for (group_name, group_id), color in zip(groups.items(), colors):
    current_data = density_data[density_data['¿Multinacional? Si/No'] == group_id]['Puntaje_Total_Sostenibilidad']
    if len(current_data) > 1:
        kde = gaussian_kde(current_data)
        x_range = np.linspace(current_data.min(), current_data.max(), 500)
        y_values = kde(x_range)
        fig12.add_trace(go.Scatter(
            x=x_range, 
            y=y_values, 
            mode='lines', 
            name=group_name,
            line=dict(color=color),
            fill='tozeroy'
        ))
fig12.update_layout(
    title_text='<b>Distribución del Desempeño: Multinacional vs. Nacional</b>',
    title_x=0.5,
    xaxis_title='Puntaje Total de Sostenibilidad',
    yaxis_title='Densidad',
    legend_title_text='Tipo de Empresa'
)
fig12.show()
save_chart_as_html(fig12, '12_density_plot_multinational_vs_national.html')
fig12.write_image("img/12_density_plot_multinational_vs_national.png", width=1000, height=800, scale=2)

# 13. Análisis de Foco Temático: Comparativa de Variables entre Líderes y Rezagados
var_importance_data = raw_data.groupby(['Macrosector', 'Variable'])['valoracionPonderada'].mean().reset_index()
var_importance_matrix = var_importance_data.pivot(
    index='Variable', 
    columns='Macrosector', 
    values='valoracionPonderada'
).fillna(0)

fig13 = px.imshow(
    var_importance_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale='Blues',
    labels=dict(x="Macrosector", y="Variable de Sostenibilidad", color="Importancia Promedio")
)
fig13.update_layout(
    title_text='<b>Importancia Relativa de Variables por Macrosector</b><br><sup>El color representa la contribución promedio de cada variable al puntaje del sector</sup>',
    title_x=0.5,
    xaxis_tickangle=-45,
    height=max(600, len(var_importance_matrix.index) * 20)
)
fig13.show()
save_chart_as_html(fig13, '13_variable_importance_heatmap.html')
fig13.write_image("img/13_variable_importance_heatmap.png", width=1000, height=800, scale=2)

# 14. Gráfico de Dispersión con Línea de Tendencia: Ingresos vs. Valoración Ponderada
scatter_data = company_agg_data[company_agg_data['Ingresos_Operacionales'] > 0].copy()

fig14 = px.scatter(
    scatter_data,
    x="Ingresos_Operacionales",
    y="Puntaje_Total_Sostenibilidad",
    color="Macrosector",
    hover_name="Razón social",
    log_x=True,
    trendline="ols",
    trendline_scope="overall"
)
fig14.update_layout(
    title='<b>Relación entre Ingresos Operacionales y Desempeño en Sostenibilidad</b>',
    title_x=0.5,
    xaxis_title='Ingresos Operacionales (Escala Logarítmica)',
    yaxis_title='Puntaje Total de Sostenibilidad',
    legend_title_text='Macrosector'
)
fig14.show()
save_chart_as_html(fig14, '14_scatter_trend_income_vs_sustainability.html')
fig14.write_image("img/14_scatter_trend_income_vs_sustainability.png", width=1000, height=800, scale=2)

# 15. Diagrama de Cuerdas (Chord Diagram) de Interconexión Sector-Pilar
chord_data = raw_data.groupby(['Macrosector', 'Nombre Pilar'])['valoracionPonderada'].sum().reset_index()
chord_matrix = chord_data.pivot(
    index='Macrosector', 
    columns='Nombre Pilar', 
    values='valoracionPonderada'
).fillna(0)

fig15 = px.imshow(
    chord_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='Viridis',
    labels=dict(x="Pilar de Sostenibilidad", y="Macrosector", color="Valoración Total")
)
fig15.update_layout(
    title_text='<b>Interconexión y Especialización: Macrosector vs. Pilar de Sostenibilidad</b><br><sup>El color representa la valoración total acumulada</sup>',
    title_x=0.5,
    xaxis_tickangle=-45
)
fig15.show()
save_chart_as_html(fig15, '15_chord_alternative_macrosector_vs_pillar.html')
fig15.write_image("img/15_chord_alternative_macrosector_vs_pillar.png", width=1000, height=800, scale=2)

# 16. Gráfico de Barras Anidadas: Variables Clave por Pilar y Liderazgo Sectorial
df_agg = raw_data.groupby(['Nombre Pilar', 'Variable', 'Macrosector'])['valoracionPonderada'].mean().reset_index()
idx = df_agg.groupby(['Nombre Pilar', 'Variable'])['valoracionPonderada'].idxmax()
df_leaders = df_agg.loc[idx][['Nombre Pilar', 'Variable', 'Macrosector']]
df_leaders.rename(columns={'Macrosector': 'Sector_Lider'}, inplace=True)
df_plot_data = df_agg.groupby(['Nombre Pilar', 'Variable'])['valoracionPonderada'].mean().reset_index()
df_plot_data = pd.merge(df_plot_data, df_leaders, on=['Nombre Pilar', 'Variable'])
df_plot_data['Variable'] = df_plot_data['Variable'].apply(lambda x: wrap_text(x, max_length=30))

fig16 = px.bar(
    df_plot_data,
    x='valoracionPonderada',
    y='Variable',
    color='Sector_Lider',
    orientation='h',
    facet_col='Nombre Pilar',
    facet_col_wrap=3,
    labels={'valoracionPonderada': 'Valoración Ponderada Promedio', 'Variable': ''},
    color_discrete_sequence=px.colors.qualitative.Set2,
    facet_col_spacing=0.01
)
fig16.update_layout(
    title_text='<b>Variables Clave por Pilar y Liderazgo Sectorial</b><br><sup>El color de la barra indica el Macrosector con mayor puntaje en esa variable</sup>',
    title_x=0.5,
    height=max(800, len(df_plot_data['Nombre Pilar'].unique()) * 300),
)
fig16.update_yaxes(matches=None, showticklabels=True)
fig16.update_yaxes(categoryorder="total ascending")
fig16.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
save_chart_as_html(fig16, '16_nested_bars_variables_by_pillar_and_sector.html')
fig16.write_image("img/16_nested_bars_variables_by_pillar_and_sector.png", width=1000, height=800, scale=2)

# 17. Gráfico de Barras Anidadas: Variables Clave por Pilar y Liderazgo Sectorial
df_agg = raw_data.groupby(['Nombre Pilar', 'Variable', 'Macrosector'])['valoracionPonderada'].mean().reset_index()

idx = df_agg.groupby(['Nombre Pilar', 'Variable'])['valoracionPonderada'].idxmax()
df_leaders = df_agg.loc[idx][['Nombre Pilar', 'Variable', 'Macrosector']]
df_leaders.rename(columns={'Macrosector': 'Sector_Lider'}, inplace=True)

df_plot_data = df_agg.groupby(['Nombre Pilar', 'Variable'])['valoracionPonderada'].mean().reset_index()

df_plot_data = pd.merge(df_plot_data, df_leaders, on=['Nombre Pilar', 'Variable'])
df_plot_data['Variable'] = df_plot_data['Variable'].apply(lambda x: wrap_text(x, max_length=30))

fig_nested_scatter = px.scatter(
    df_plot_data,
    x='valoracionPonderada',
    y='Variable',
    size='valoracionPonderada',
    color='Sector_Lider',     
    hover_name='Sector_Lider',
    facet_col='Nombre Pilar',
    facet_col_wrap=3,         
    labels={'valoracionPonderada': 'Valoración Ponderada Promedio', 'Variable': ''},
        facet_col_spacing=0.15
)

fig_nested_scatter.update_layout(
    title_text='<b>Variables Clave por Pilar y Liderazgo Sectorial</b><br><sup>El color indica el Macrosector líder; el tamaño, la importancia de la variable</sup>',
    title_x=0.5,
    font=dict(family="Arial, sans-serif", size=10, color="black"),
    height=max(800, len(df_plot_data['Nombre Pilar'].unique()) * 200),

)
fig_nested_scatter.update_yaxes(matches=None, showticklabels=True)
fig_nested_scatter.update_yaxes(categoryorder="total ascending")
fig_nested_scatter.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig_nested_scatter.show()
save_chart_as_html(fig_nested_scatter, '17_nested_scatter_variables_by_pillar_and_sector.html')
fig_nested_scatter.write_image("img/17_nested_scatter_variables_by_pillar_and_sector.png", width=1000, height=800, scale=2)

