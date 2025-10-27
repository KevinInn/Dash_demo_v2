import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash import html
import dash_bootstrap_components as dbc
def build_compare_figure(df_result, chart_type, title):
    metric_columns = [col for col in df_result.columns if col != 'Country']
    fig = go.Figure()

    if not metric_columns:
        fig.update_layout(
            template='plotly_dark', font=dict(color='#deb522'), title=title,
            annotations=[dict(text='沒有可比較的指標', x=0.5, y=0.5, showarrow=False, font=dict(color='#deb522'))]
        )
        return fig

    df_numeric = df_result.copy()
    for col in metric_columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    if chart_type == 'radar':
        df_normalized = df_numeric.copy()
        for col in metric_columns:
            series = df_numeric[col]
            if series.dropna().empty:
                df_normalized[col] = np.nan
                continue
            min_val, max_val = series.min(), series.max()
            if max_val > min_val:
                df_normalized[col] = 100 * (series - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 50

        for _, row in df_normalized.iterrows():
            values = [row[col] if pd.notna(row[col]) else 0 for col in metric_columns]
            if values:
                values.append(values[0])
            theta = metric_columns + [metric_columns[0]] if metric_columns else metric_columns
            fig.add_trace(go.Scatterpolar(r=values, theta=theta, fill='toself', name=row['Country']))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template='plotly_dark', font=dict(color='#deb522'), title=title, height=600
        )

    elif chart_type == 'bar':
        for col in metric_columns:
            fig.add_trace(go.Bar(
                name=col, x=df_result['Country'], y=df_numeric[col],
                text=df_numeric[col].round(2), textposition='auto'
            ))
        fig.update_layout(
            barmode='group', template='plotly_dark', font=dict(color='#deb522'), title=title,
            xaxis_title='Country', yaxis_title='Value', height=600
        )

    else:  # line
        for col in metric_columns:
            fig.add_trace(go.Scatter(
                x=df_result['Country'], y=df_numeric[col], mode='lines+markers+text',
                name=col, text=df_numeric[col].round(2), textposition='top center'
            ))
        fig.update_layout(
            template='plotly_dark', font=dict(color='#deb522'), title=title,
            xaxis_title='Country', yaxis_title='Value', height=600
        )

    return fig

def generate_stats_card(title, value, image_path):
    return html.Div(
        dbc.Card([
            dbc.CardImg(src=image_path, top=True, style={'width': '50px', 'height': '50px','alignSelf': 'center'}),
            dbc.CardBody([
                html.P(value, className="card-value",
                       style={'margin': '0px','fontSize': '22px','fontWeight': 'bold'}),
                html.H4(title, className="card-title",
                        style={'margin': '0px','fontSize': '18px','fontWeight': 'bold'})
            ], style={'textAlign': 'center'}),
        ], style={'paddingBlock':'10px',"backgroundColor":'#deb522','border':'none','borderRadius':'10px'})
    )