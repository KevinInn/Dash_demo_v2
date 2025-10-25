from dash import Dash, html, dcc, Input, State, Output, dash_table, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_leaflet as dl
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
# 導入其他模組中的函數
from src.const import get_constants
from src.generate_visualization import generate_bar, generate_pie, generate_map, generate_box
from src.data_clean import travel_data_clean, countryinfo_data_clean, data_merge

# 加載欲分析的資料集  
travel_df = pd.read_csv('./data/Travel_dataset.csv') # 旅遊資訊
country_info_df = pd.read_csv('./data/country_info.csv') # 國家資訊
attractions_df = pd.read_csv('./data/Attractions.csv') # 景點資訊

# 進行資料前處理
travel_df = travel_data_clean(travel_df)
country_info_df = countryinfo_data_clean(country_info_df)

# 合併travel_df和country_info_df，方便後續分析
df_merged = data_merge(travel_df, country_info_df)

# 獲取國家名稱列表
country_list = list(attractions_df['country'].unique())

# 切換頁面（如有需要可以自行增加）
def load_data(tab):
    if tab == 'travel' or tab == 'planner':
        return df_merged
    # elif tab == 'other': # 可以自行增加其他頁面，'other'為頁面名稱，可自行更改設定名稱
    #     return other_df # 此頁面要顯示的資料集

# 呼叫 ./src/const.py 中的 get_constants() 函式
# 獲取統計數據(畫面上方的四格數據:目的地國家數、旅遊者數、旅遊者國籍數、平均旅遊天數)
num_of_country, num_of_traveler, num_of_nationality, avg_days = get_constants(travel_df)

# 初始化應用程式
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='Travel Data Analysis Dashboard', suppress_callback_exceptions=True)
server = app.server

# 生成統計數據的顯示卡片區塊
def generate_stats_card (title, value, image_path):
    return html.Div(
        dbc.Card([ 
            dbc.CardImg(src=image_path, top=True, style={'width': '50px', 'height': '50px','alignSelf': 'center'}), # icon圖片外觀設定
            dbc.CardBody([
                html.P(value, className="card-value", style={'margin': '0px','fontSize': '22px','fontWeight': 'bold'}), # 數據外觀設定
                html.H4(title, className="card-title", style={'margin': '0px','fontSize': '18px','fontWeight': 'bold'}) # 標題外觀設定
            ], style={'textAlign': 'center'}),
        ], style={'paddingBlock':'10px',"backgroundColor":'#deb522','border':'none','borderRadius':'10px'}) # 卡片外觀設定
    )

# 外觀設定
tab_style = {
    'idle':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'backgroundColor': '#deb522',
        'border':'none'
    },
    'active':{
        'borderRadius': '10px',
        'padding': '0px',
        'marginInline': '5px',
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'center',
        'fontWeight': 'bold',
        'border':'none',
        'textDecoration': 'underline',
        'backgroundColor': '#deb522'
    }
}

MAX_OPTIONS_DISPLAY = 3300

COMPARE_METRIC_OPTIONS = [
    {'label': ' 安全指數 (Safety Index)', 'value': 'safety'},
    {'label': ' 消費物價指數 (CPI)', 'value': 'cpi'},
    {'label': ' 個人消費支出 (PCE)', 'value': 'pce'},
    {'label': ' 平均住宿成本', 'value': 'accommodation'},
    {'label': ' 平均交通成本', 'value': 'transportation'},
    {'label': ' 遊客人數統計', 'value': 'travelers'},
]

DEFAULT_COMPARE_METRICS = ['safety', 'cpi', 'accommodation']
MAX_COMPARE_COUNTRIES = 5


def prepare_country_compare_data(countries, metrics):
    valid_metrics = metrics or []
    valid_countries = countries or []

    if not valid_countries or not valid_metrics:
        return pd.DataFrame(), []

    # 去除非字串或重複國家，並限制最大數量
    seen = set()
    deduped = []
    for country in valid_countries:
        if not isinstance(country, str):
            continue
        if country in seen:
            continue
        seen.add(country)
        deduped.append(country)

    available_destinations = set(df_merged['Destination'].dropna().unique())
    limited_countries = [c for c in deduped if c in available_destinations][:MAX_COMPARE_COUNTRIES]

    if not limited_countries:
        return pd.DataFrame(), []

    df_compare = df_merged[df_merged['Destination'].isin(limited_countries)].copy()
    if df_compare.empty:
        return pd.DataFrame(), []

    comparison_data = []
    for country in limited_countries:
        country_data = df_compare[df_compare['Destination'] == country]
        if country_data.empty:
            continue

        row = {'Country': country}

        if 'safety' in valid_metrics and 'Safety Index' in country_data.columns:
            safety = country_data['Safety Index'].dropna()
            row['Safety Index'] = safety.iloc[0] if not safety.empty else np.nan

        if 'cpi' in valid_metrics and 'CPI' in country_data.columns:
            cpi = country_data['CPI'].dropna()
            row['CPI'] = cpi.iloc[0] if not cpi.empty else np.nan

        if 'pce' in valid_metrics and 'PCE' in country_data.columns:
            pce = country_data['PCE'].dropna()
            row['PCE'] = pce.iloc[0] if not pce.empty else np.nan

        if 'accommodation' in valid_metrics and 'Accommodation cost' in country_data.columns:
            acc_cost = pd.to_numeric(country_data['Accommodation cost'], errors='coerce')
            row['Avg Accommodation Cost'] = acc_cost.mean() if not acc_cost.isna().all() else np.nan

        if 'transportation' in valid_metrics and 'Transportation cost' in country_data.columns:
            trans_cost = pd.to_numeric(country_data['Transportation cost'], errors='coerce')
            row['Avg Transportation Cost'] = trans_cost.mean() if not trans_cost.isna().all() else np.nan

        if 'travelers' in valid_metrics:
            row['Total Travelers'] = len(country_data)

        comparison_data.append(row)

    df_result = pd.DataFrame(comparison_data)
    return df_result, limited_countries


def build_compare_figure(df_result, chart_type, title):
    metric_columns = [col for col in df_result.columns if col != 'Country']

    fig = go.Figure()
    if not metric_columns:
        fig.update_layout(
            template='plotly_dark',
            font=dict(color='#deb522'),
            title=title,
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
            template='plotly_dark',
            font=dict(color='#deb522'),
            title=title,
            height=600
        )

    elif chart_type == 'bar':
        for col in metric_columns:
            fig.add_trace(go.Bar(
                name=col,
                x=df_result['Country'],
                y=df_numeric[col],
                text=df_numeric[col].round(2),
                textposition='auto'
            ))

        fig.update_layout(
            barmode='group',
            template='plotly_dark',
            font=dict(color='#deb522'),
            title=title,
            xaxis_title='Country',
            yaxis_title='Value',
            height=600
        )

    else:  # line chart
        for col in metric_columns:
            fig.add_trace(go.Scatter(
                x=df_result['Country'],
                y=df_numeric[col],
                mode='lines+markers+text',
                name=col,
                text=df_numeric[col].round(2),
                textposition='top center'
            ))

        fig.update_layout(
            template='plotly_dark',
            font=dict(color='#deb522'),
            title=title,
            xaxis_title='Country',
            yaxis_title='Value',
            height=600
        )

    return fig


def create_compare_table_component(df_result):
    if df_result.empty:
        return html.Div('沒有可顯示的比較數據。', style={'color': 'white'})

    df_display = df_result.copy()
    for col in df_display.columns:
        if col == 'Country':
            continue
        if col == 'Total Travelers':
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce').fillna(0).astype(int).astype(str)
        else:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce').apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A'
            )

    return dash_table.DataTable(
        data=df_display.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df_display.columns],
        style_data={'backgroundColor': '#deb522', 'color': 'black'},
        style_header={'backgroundColor': 'black', 'color': '#deb522', 'fontWeight': 'bold'},
        style_table={'overflowX': 'auto'},
        export_format='csv'
    )

# 定義系統的佈局
app.layout = html.Div([
    dbc.Container([
        # 頂部Logo與切換頁面 (Overview/Attractions)
        dbc.Row([
            # 左方Logo區塊
            dbc.Col(html.Img(src="./assets/logo.png", height=100), width=5, style={'marginTop': '15px'}),
            
            # 右方Overview/Attractions切換區塊
            # 可參考簡報 Callback function ( tabs.py )
            dbc.Col(
                dcc.Tabs(id='graph-tabs', value='overview', children=[
                    dcc.Tab(label='Overview', value='overview',style=tab_style['idle'],selected_style=tab_style['active']),
                    dcc.Tab(label='Trip Planner', value='planner', style=tab_style['idle'], selected_style=tab_style['active']),
                    dcc.Tab(label='Attractions', value='attractions',style=tab_style['idle'],selected_style=tab_style['active']),

                    # 若有其他頁面可以自行增加
                    # dcc.Tab(label='Other Page', value='other_page',style=tab_style['idle'],selected_style=tab_style['active']),
                ], style={'height':'50px'})
            ,width=7, style={'alignSelf': 'center'}),
        ]),

        # 統計數據：Country、Traveler、Nationality、Average Days
        dbc.Row([
            dbc.Col(generate_stats_card("Country", num_of_country, "./assets/earth.svg"), width=3),
            dbc.Col(generate_stats_card("Traveler", num_of_traveler, "./assets/user.svg"), width=3),
            dbc.Col(generate_stats_card("Nationality", num_of_nationality, "./assets/earth.svg"), width=3),
            dbc.Col(generate_stats_card("Average Days", avg_days, "./assets/calendar.svg"), width=3),
        ],style={'marginBlock': '10px'}),
        
        # TODO: 為何要這個?? 中間切換頁面
        dbc.Row([
            dcc.Tabs(id='tabs', value='travel', children=[
                dcc.Tab(label='Travel Data', value='travel',style={'border':'1px line white','backgroundColor':'black','color': '#deb522','fontWeight': 'bold'},selected_style={'border':'1px solid white','backgroundColor':'black','color': '#deb522','fontWeight': 'bold','textDecoration': 'underline'}),
                # 若有其他頁面可以自行增加
                # dcc.Tab(label='Other', value='other',style={'border':'1px solid white','backgroundColor':'black','color': '#deb522','fontWeight': 'bold'},selected_style={'border':'1px solid white','backgroundColor':'black','color': '#deb522','fontWeight': 'bold','textDecoration': 'underline'}),
            ], style={'padding': '0px'})
        ]),

        # 用於顯示不同頁面的內容
        html.Div(id='graph-content')

    ], style={'padding': '0px'})
], style={'backgroundColor': 'black', 'minHeight': '100vh'})


# 根據選擇的標籤頁更新顯示的內容
@app.callback(
    Output('graph-content', 'children'), # callback function output: id為'graph-content'的 children（第119行程式碼）
    [Input('graph-tabs', 'value')] # callback function input: id為'graph-tabs'的 value值（第92行程式碼）
)
def render_tab_content(tab): # 針對上述的input值要做的處理，tab = Input('graph-tabs', 'value')
    if tab == 'overview':
        # 返回 'Overview' 頁面的佈局
        return html.Div([
            # 第一排下拉選單 - 長條圖(Col1) & 圓餅圖(Col2)
            # 可參考簡報 Callback function ( dropdown.py )
            # 可參考簡報 資料視覺化( visualizing.py )
            dbc.Row([
                dbc.Col([
                    html.H3("各大洲或各國不同月份遊客人数", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-bar-1',
                        options=[
                            {'label': i, 'value': i} for i in pd.concat([df_merged['Continent'], df_merged['Destination']]).unique()
                        ],
                        placeholder='Select a continent or country',
                        style={'width': '90%', 'margin-top': '10px', 'margin-bottom': '10px'}
                    )
                ]),
                dbc.Col([
                    html.H3("各大洲或各國的遊客屬性、住宿及交通類型", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-pie-1',
                        options = [
                            {'label': i, 'value': i} for i in pd.concat([df_merged['Continent'], df_merged['Destination']]).unique()
                        ],
                        placeholder='Select a continent or country',
                        style={'width': '50%', 'margin-top': '5px', 'margin-bottom': '5px', 'display': 'inline-block'}
                    ),
                    dcc.Dropdown(
                        id='dropdown-pie-2',
                        options = [
                            {'label': i, 'value': i} for i in ['Traveler nationality', 'Age group', 'Traveler gender', 'Accommodation type', 'Transportation type']
                        ],
                        placeholder='Select a value',
                        style={'width': '50%', 'margin-top': '5px', 'margin-bottom': '5px', 'display': 'inline-block'}
                    )
                ]),
            ]),
            # 第一排圖表顯示區 - 長條圖(Col1：tabs-content-1) & 圓餅圖(Col2：tabs-content-2)
            dbc.Row([
                dbc.Col([
                    dcc.Loading([
                        html.Div(id='tabs-content-1'),
                    ],
                    type='default',color='#deb522'),
                ]),
                dbc.Col([
                    dcc.Loading([
                        html.Div(id='tabs-content-2'),
                    ],
                    type='default',color='#deb522'),
                ]),
            ]),
            # 第二排下拉選單 - 地圖(Col1) & 箱型圖(Col2)
            dbc.Row([
                dbc.Col([
                    html.H3("各大洲或各國安全係數及消費水平", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-map-1',
                        options = [{'label': 'All', 'value': None}] 
                                    + [{'label': i, 'value': i} for i in df_merged['Continent'].unique()],
                        placeholder='Select a continent or country',
                        style={'width': '50%', 'margin-top': '5px', 'margin-bottom': '5px', 'display': 'inline-block'}
                    ),
                    dcc.Dropdown(
                        id='dropdown-map-2',
                        options = [
                            {'label': i, 'value': i} for i in ['Safety Index', 'Crime_index', 'CPI', 'PCE', 'Exchange_rate']
                        ],
                        placeholder='Select a value',
                        style={'width': '50%', 'margin-top': '5px', 'margin-bottom': '5px', 'display': 'inline-block'}
                    )
                ]),
                dbc.Col([
                    html.H3("各大洲或各國家住宿及交通成本", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-box-1',
                        options = [{'label': i, 'value': i} for i in pd.concat([df_merged['Continent'], df_merged['Destination']]).unique()],           
                        placeholder='Select a continent or country',
                        style={'width': '50%', 'margin-top': '5px', 'margin-bottom': '5px', 'display': 'inline-block'}
                    ),
                    dcc.Dropdown(
                        id='dropdown-box-2',
                        options = [
                            {'label': i, 'value': i} for i in ['Accommodation cost', 'Transportation cost']
                        ],
                        placeholder='Select a value',
                        style={'width': '50%', 'margin-top': '5px', 'margin-bottom': '5px', 'display': 'inline-block'}
                    )
                ]),
            ]),
            # 第二排圖表顯示區 - 地圖(Col1：tabs-content-3) & 箱型圖(Col2：tabs-content-4)
            dbc.Row([
                dbc.Col([
                    dcc.Loading([
                        html.Div(id='tabs-content-3'),
                    ],
                    type='default',color='#deb522'),
                ]),
                dbc.Col([
                    dcc.Loading([
                        html.Div(id='tabs-content-4'),
                    ],
                    type='default',color='#deb522'),
                ]),
            ]),
        ])
    # 可參考簡報 Callback function (multiValueDropdown.py )
    elif tab == 'attractions':
        return html.Div([
            dcc.Dropdown(
                options=[{'label': country, 'value': country} for country in country_list],
                value='Australia',  # 單一值，不是 list
                id='attractions-dropdown',
                multi=False,  # 單選
                style={'backgroundColor': '#deb522', 'color': 'black'}
            ),
            html.Button(
                "查詢",
                id='attractions-submit',
                n_clicks=0,
                className="btn btn-primary",
                style={
                    'backgroundColor': '#deb522',
                    'color': 'black',
                    'fontWeight': 'bold',
                    'marginTop': '10px',
                    'padding': '6px 16px',
                    'borderRadius': '6px',
                    'border': 'none',
                    'cursor': 'pointer'
                }
            ),
            dcc.Loading(
                id="attractions-loading",
                type="circle",  # 可改成 "default" / "cube" / "dot"
                color="#deb522",
                fullscreen=False,  # 若想覆蓋全頁可設 True
                children=[
                    html.Div(id='attractions-output-container', style={'overflow-x': 'auto', 'marginTop': '10px'}),
                    html.Div(id='attractions-map-container', style={'height': '600px', 'marginTop': '16px'})
                ]
            )
        ])

    elif tab == 'planner':
        accommodation_types = sorted(travel_df['Accommodation type'].dropna().unique().tolist())

        # 從「實際資料」動態取得所有 Travel Alert 值（聯集）
        # 以 df_merged（travel + country_info merge 結果）為主，避免漏國家層欄位
        # —— 新版：從 country_info_df 與 df_merged 取聯集，避免漏色（例如「橙色」）——
        if 'Travel Alert' in country_info_df.columns:
            alerts_from_country = country_info_df['Travel Alert'].dropna().astype(str).str.strip().tolist()
        else:
            alerts_from_country = []

        if 'Travel Alert' in df_merged.columns:
            alerts_from_merged = df_merged['Travel Alert'].dropna().astype(str).str.strip().tolist()
        else:
            alerts_from_merged = []

        seen_alerts = sorted(set(alerts_from_country) | set(alerts_from_merged))

        # 風險排序（安全→危險）。未知值給中位順位，僅用來讓選單從低到高排得直覺。
        alert_rank_map = {'綠色': 0, '藍色': 1, '灰色': 2, '黃色': 3, '橙色': 4, '紅色': 5, '黑色': 6}
        default_rank = 3
        def alert_sort_key(c): 
            return alert_rank_map.get(c, default_rank)

        color_options = [{'label': c, 'value': c} for c in sorted(seen_alerts, key=alert_sort_key)]
        default_alert = ('灰色' if '灰色' in seen_alerts else (color_options[0]['value'] if color_options else None))

        return html.Div([
            dcc.Store(id='planner-selected-countries', data=[]),
            html.H3("Trip Planner：用預算、安全與住宿偏好找國家", style={'color': '#deb522', 'margin-top': '5px'}),

            # ===== 篩選列 1：住宿費用 & 住宿類型 =====
            dbc.Row([
                dbc.Col([
                    html.Label("Accommodation cost（min）", style={'color': '#deb522'}),
                    dcc.Input(id='planner-cost-min', type='number', placeholder='min',
                              style={'width': '100%', 'backgroundColor': 'black', 'color': '#deb522', 'border': '1px solid #deb522'})
                ], width=3),
                dbc.Col([
                    html.Label("Accommodation cost（max）", style={'color': '#deb522'}),
                    dcc.Input(id='planner-cost-max', type='number', placeholder='max',
                              style={'width': '100%', 'backgroundColor': 'black', 'color': '#deb522', 'border': '1px solid #deb522'})
                ], width=3),
                dbc.Col([
                    html.Label("Accommodation type（multi）", style={'color': '#deb522'}),
                    dcc.Dropdown(id='planner-acc-types',
                                 options=[{'label': t, 'value': t} for t in accommodation_types],
                                 value=[], multi=True,
                                 style={'backgroundColor': '#deb522', 'color': 'black'})
                ], width=6),
            ], style={'marginTop': '8px', 'marginBottom': '12px'}),

            # ===== 篩選列 2：安全顏色門檻、免簽 =====
            dbc.Row([
                dbc.Col([
                    html.Label("可接受的最高危險顏色（含以下等級）", style={'color': '#deb522'}),
                    dcc.Dropdown(
                        id='planner-alert-max',
                        options=color_options,
                        value=default_alert,
                        clearable=False,
                        style={'backgroundColor': '#deb522', 'color': 'black'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Visa", style={'color': '#deb522'}),
                    dcc.Checklist(
                        id='planner-visa-only',
                        options=[{'label': ' 只顯示免簽', 'value': 'exempt'}],
                        value=[],
                        inputStyle={'marginRight': '6px'},
                        labelStyle={'color': '#deb522'}
                    )
                ], width=6),
            ], style={'marginBottom': '12px'}),

            # ===== 權重（只留 Safety / Cost；CPI 內嵌到成本換算）=====
            dbc.Row([
                dbc.Col([
                    html.Label("Weights（0–10）：Safety / Cost", style={'color': '#deb522'}),
                    html.Div([
                        dcc.Slider(id='w-safety', min=0, max=10, step=1, value=7, marks=None, tooltip={'always_visible': True}),
                        dcc.Slider(id='w-cost', min=0, max=10, step=1, value=8, marks=None, tooltip={'always_visible': True}),
                    ], style={'paddingTop': '10px'})
                ], width=12),
            ], style={'marginBottom': '8px'}),

            dcc.Loading([html.Div(id='planner-table-container')], type='default', color='#deb522'),

            html.Hr(style={'borderColor': '#deb522'}),
            html.H4("篩選國家即時比較", style={'color': '#deb522', 'marginTop': '10px'}),
            html.Small("自動套用符合條件的前 5 個國家至三種比較圖表。", style={'color': '#deb522'}),

            dbc.Row([
                dbc.Col([
                    html.Label("預設套用排行前五，可自行調整比較國家（最多5個）", style={'color': '#deb522', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='planner-compare-country-selector',
                        options=[],
                        value=[],
                        multi=True,
                        placeholder='預設顯示前五個建議國家，可透過此處增減。',
                        style={'backgroundColor': '#deb522', 'color': 'black'}
                    )
                ], width=12)
            ], style={'marginTop': '12px', 'marginBottom': '12px'}),

            dbc.Row([
                dbc.Col([
                    html.Label("選擇比較指標", style={'color': '#deb522', 'fontWeight': 'bold'}),
                    dcc.Checklist(
                        id='planner-compare-metrics',
                        options=COMPARE_METRIC_OPTIONS,
                        value=DEFAULT_COMPARE_METRICS,
                        inputStyle={'marginRight': '6px'},
                        labelStyle={'color': '#deb522', 'display': 'block', 'marginBottom': '8px'}
                    )
                ], width=12),
            ], style={'marginTop': '12px', 'marginBottom': '12px'}),

            dbc.Row([
                dbc.Col([
                    dcc.Loading(html.Div(id='planner-compare-radar'), type='default', color='#deb522')
                ], width=6),
                dbc.Col([
                    dcc.Loading(html.Div(id='planner-compare-bar'), type='default', color='#deb522')
                ], width=6),
            ], style={'marginBottom': '12px'}),

            dbc.Row([
                dbc.Col([
                    dcc.Loading(html.Div(id='planner-compare-line'), type='default', color='#deb522')
                ], width=12),
            ], style={'marginBottom': '12px'}),

            html.H4("詳細數據對比", style={'color': '#deb522', 'marginTop': '10px'}),
            dcc.Loading([
                html.Div(id='planner-compare-table-container')
            ], type='default', color='#deb522')
        ])

    else:
        return html.Div("選擇的標籤頁不存在。", style={'color': 'white'})

# 可參考簡報Callback function ( dropdown.py )
# 長條圖回調函數
@app.callback(
    Output('tabs-content-1', 'children'), # callback function output: id為'tabs-content-1'的 children（第173行程式碼）
    [Input('dropdown-bar-1', 'value'),  # callback function input 1: id為'dropdown-bar-1'的 value值（第141行程式碼）
     Input('graph-tabs', 'value')]  # callback function input 2: id為'graph-tabs'的 value值（第92行程式碼）
)
# 更新長條圖
def update_bar_chart(dropdown_value, tab): # 針對上述的input值要做的處理，dropdown_value = Input('dropdown-bar-1', 'value')，tab = Input('graph-tabs', 'value')
    
    # 如果當前頁面不是選擇'overview'，則不更新圖表
    if tab != 'overview':
        return no_update
    
    # 選取要用的資料(第26行程式碼的function)
    df = load_data('travel')

    # 生成長條圖
    fig1 = generate_bar(df, dropdown_value)

    # 回傳包含長條圖的html.Div
    return html.Div([
        dcc.Graph(id='graph1', figure=fig1),
    ], style={'width': '90%', 'display': 'inline-block'})

# 圓餅圖回調
@app.callback(
    Output('tabs-content-2', 'children'), # callback function output: id為'tabs-content-2'的 children（第179行程式碼） 
    [Input('dropdown-pie-1', 'value'), # callback function input 1: id為'dropdown-pie-1'的 value值（第145行程式碼）TODO: 改行數
     Input('dropdown-pie-2', 'value'), # callback function input 2: id為'dropdown-pie-2'的 value值（第153行程式碼）
     Input('graph-tabs', 'value')] # callback function input 2: id為'graph-tabs'的 value值（第87行程式碼）
)
# 更新圓餅圖
def update_pie_chart(dropdown_value_1, dropdown_value_2, tab): # 針對上述的input值要做的處理，dropdown_value_1 = Input('dropdown-pie-1', 'value')，dropdown_value_2 = Input('dropdown-pie-2', 'value')，tab = Input('graph-tabs', 'value')
    # 如果當前頁面不是選擇'overview'，則不更新圖表
    if tab != 'overview':
        return no_update

    # 選取要用的資料(第26行程式碼的function)
    df = load_data('travel')

    # 生成圓餅圖
    fig2 = generate_pie(df, dropdown_value_1, dropdown_value_2)
    
    # 回傳包含圓餅圖的html.Div
    return html.Div([
        dcc.Graph(id='graph2', figure=fig2),
    ], style={'width': '90%', 'display': 'inline-block'})

# 地圖回調
@app.callback(
    Output('tabs-content-3', 'children'), # callback function output: id為'tabs-content-3'的 children（第219行程式碼）
    [Input('dropdown-map-1', 'value'), # callback function input 1: id為'dropdown-map-1'的 value值（第182行程式碼）
     Input('dropdown-map-2', 'value'), # callback function input 2: id為'dropdown-map-2'的 value值（第189行程式碼）
     Input('graph-tabs', 'value')] # callback function input 3: id為'graph-tabs'的 value值（第87行程式碼）
)
# 更新地圖
def update_map(dropdown_value_1, dropdown_value_2, tab): # 針對上述的input值要做的處理，dropdown_value_1 = Input('dropdown-map-1', 'value')，dropdown_value_2 = Input('dropdown-map-2', 'value')，tab = Input('graph-tabs', 'value')
    # 如果當前頁面不是選擇'overview'，則不更新圖表
    if tab != 'overview':
        return no_update

    # 選取要用的資料(第26行程式碼的function)
    df = load_data('travel')

    # 生成地圖
    fig3 = generate_map(df, dropdown_value_1, dropdown_value_2)
    
    # 回傳包含地圖的html.Div
    return html.Div([
        dcc.Graph(id='graph3', figure=fig3),
    ], style={'width': '90%', 'display': 'inline-block'})

# 箱型圖回調
@app.callback(
    Output('tabs-content-4', 'children'), # callback function output: id為'tabs-content-4'的 children（第227行程式碼）
    [Input('dropdown-box-1', 'value'), # callback function input 1: id為'dropdown-box-1'的 value值（第202行程式碼）
     Input('dropdown-box-2', 'value'), # callback function input 2: id為'dropdown-box-2'的 value值（第208行程式碼）
     Input('graph-tabs', 'value')] # callback function input 3: id為'graph-tabs'的 value值（第87行程式碼）
)
# 更新箱型圖
def update_box_chart(dropdown_value_1, dropdown_value_2, tab): # 針對上述的input值要做的處理，dropdown_value_1 = Input('dropdown-box-1', 'value')，dropdown_value_2 = Input('dropdown-box-2', 'value')，tab = Input('graph-tabs', 'value')
    # 如果當前頁面不是選擇'overview'，則不更新圖表
    if tab != 'overview':
        return no_update
    
    # 選取要用的資料(第26行程式碼的function)
    df = load_data('travel')

    # 生成箱型圖
    fig4 = generate_box(df, dropdown_value_1, dropdown_value_2)
    
    # 回傳包含箱型圖的html.Div
    return html.Div([
        dcc.Graph(id='graph4', figure=fig4),
    ], style={'width': '90%', 'display': 'inline-block'})

# 景點下拉式選單回調
@app.callback(
    [Output('attractions-output-container', 'children'),
     Output('attractions-map-container', 'children')],
    [Input('attractions-submit', 'n_clicks'),
     Input('graph-tabs', 'value')],
    [State('attractions-dropdown', 'value')],
    prevent_initial_call=True
)
def update_attractions_output(n_clicks, tab, chosen_country):
    if tab != 'attractions':
        raise PreventUpdate

    if n_clicks == 0 or not chosen_country:
        return (
            html.Div("請選擇一個國家並按下查詢。", style={'color': 'white'}),
            no_update
        )

    # --- 過濾景點 ---
    chosen_df = attractions_df[attractions_df['country'] == chosen_country].copy()

    # 表格（沿用你原本樣式）
    table = dash_table.DataTable(
        data=chosen_df.to_dict('records'),
        page_size=10,
        style_data={'backgroundColor': '#deb522', 'color': 'black'},
        style_header={'backgroundColor': 'black', 'color': '#deb522', 'fontWeight': 'bold'}
    )

    # --- 找欄位 ---
    def pick_col(cands):
        for c in cands:
            if c in chosen_df.columns:
                return c
        return None
    name_col = pick_col(['attraction', 'Attraction'])
    if not name_col:
        # 若找不到名稱欄位，就隨便挑第一個 column 試著當作景點名稱
        name_col = chosen_df.columns[0]

    # --- 用 Nominatim 免費查地點座標 ---
    geolocator = Nominatim(user_agent="my_dash_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)  # 避免頻率太高被拒

    points = []
    for _, r in chosen_df.iterrows():
        name = str(r[name_col])
        try:
            location = geocode(name)
            if location:
                points.append({'name': name, 'lat': location.latitude, 'lng': location.longitude})
        except Exception as e:
            continue

    if not points:
        return table, html.Div("選定國家目前沒有可用座標的景點。", style={'color': 'white'})

    # --- Leaflet layers & markers ---
    # OSM 瓦片（免金鑰），注意 attribution 必填
    tile_layer = dl.TileLayer(
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )

    # 建立 marker 清單（點擊顯示名稱）
    markers = [
        dl.Marker(position=[p['lat'], p['lng']], children=dl.Tooltip(p['name']))
        for p in points
    ]

    # 自動計算 bounds（以便 fitBounds）
    lats = [p['lat'] for p in points]
    lngs = [p['lng'] for p in points]
    south, west = min(lats), min(lngs)
    north, east = max(lats), max(lngs)
    bounds = [[south, west], [north, east]]

    # 如果只有一個點，就用預設中心與較大 zoom
    if len(points) == 1:
        center = [points[0]['lat'], points[0]['lng']]
        zoom = 10
        the_map = dl.Map(
            id=f"map-{hash(str(bounds))}",
            children=[tile_layer, dl.LayerGroup(markers)],
            center=center, zoom=zoom, style={'width': '100%', 'height': '600px'}
        )
    else:
        the_map = dl.Map(
            id=f"map-{hash(str(bounds))}",
            children=[tile_layer, dl.LayerGroup(markers)],
            bounds=bounds,
            style={'width': '100%', 'height': '600px'}
        )

    return table, the_map


@app.callback(
    [Output('planner-table-container', 'children'),
     Output('planner-compare-country-selector', 'options'),
     Output('planner-compare-country-selector', 'value')],
    [
        Input('planner-cost-min', 'value'),
        Input('planner-cost-max', 'value'),
        Input('planner-acc-types', 'value'),

        Input('planner-alert-max', 'value'),
        Input('planner-visa-only', 'value'),

        Input('w-safety', 'value'),
        Input('w-cost', 'value'),

        Input('graph-tabs', 'value'),
    ]
)
def update_trip_planner_table(cost_min, cost_max, acc_types,
                              alert_max, visa_only,
                              w_safety, w_cost,
                              tab):
    if tab != 'planner':
        return no_update, no_update, no_update

    df_travel = travel_df.copy()

    # --- 防呆：住宿成本 min/max 對調 ---
    if cost_min is not None and cost_max is not None and cost_min > cost_max:
        cost_min, cost_max = cost_max, cost_min

    # === 預處理 ===
    df_travel['Accommodation cost'] = pd.to_numeric(df_travel['Accommodation cost'], errors='coerce').fillna(0)
    df_travel['Duration (days)'] = pd.to_numeric(df_travel['Duration (days)'], errors='coerce')
    df_travel = df_travel.dropna(subset=['Duration (days)'])
    df_travel = df_travel[df_travel['Duration (days)'] > 0]

    # 僅住宿成本（以及每日住宿成本）
    df_travel['acc_trip_cost']  = df_travel['Accommodation cost']
    df_travel['acc_daily_cost'] = df_travel['acc_trip_cost'] / df_travel['Duration (days)']

    # === 條件 1：住宿成本區間（針對「住宿成本」）===
    if cost_min is not None:
        df_travel = df_travel[df_travel['Accommodation cost'] >= float(cost_min)]
    if cost_max is not None:
        df_travel = df_travel[df_travel['Accommodation cost'] <= float(cost_max)]

    # === 條件 2：住宿類型多選 ===
    if acc_types:
        df_travel = df_travel[df_travel['Accommodation type'].isin(acc_types)]

    if df_travel.empty:
        return html.Div("沒有符合條件的國家。", style={'color': 'white'}), [], []

    # === 目的地清單 ===
    matched_countries = sorted(df_travel['Destination'].dropna().unique().tolist())

    # === 取國家層級資訊（含 Travel Alert / CPI / PCE / Safety / Visa） ===
    # 取國家層資訊（含 Travel Alert / CPI / PCE / Safety / Visa），
    # 改成「每個 Destination 取第一個非空值」，避免 drop_duplicates 選到全空的那列。
    cols_needed = ['Destination', 'CPI', 'PCE', 'Safety Index', 'Visa_exempt_entry', 'Travel Alert']
    avail_cols = [c for c in cols_needed if c in df_merged.columns]

    sub = df_merged[df_merged['Destination'].isin(matched_countries)][avail_cols].copy()

    print("#################################")
    print(sub[sub["Destination"]=="South African"])
    print("#################################")
    
    def first_nonnull(series):
        for v in series:
            # 去掉純空白字串
            if pd.notna(v) and str(v).strip() != '':
                return v
        return np.nan

    agg_spec = {col: first_nonnull for col in avail_cols if col != 'Destination'}

    df_country = (
        sub
        .groupby('Destination', as_index=False)
        .agg(agg_spec)
    )
    key_cols = ['CPI', 'PCE', 'Safety Index', 'Travel Alert']
    df_country = df_country.dropna(subset=key_cols, how='any')
    print("==========================")
    print(df_country)
    print("==========================")
    # === Travel Alert 門檻（同等或更安全） ===
    # 排序映射（未知值置中）
    alert_rank_map = {'綠色': 0, '藍色': 1, '灰色': 2, '黃色': 3, '橙色': 4, '紅色': 5, '黑色': 6}
    default_rank = 3
    def alert_rank(v):
        return alert_rank_map.get(str(v).strip(), default_rank)
    if alert_max is not None:
        max_rank = alert_rank(alert_max)
        df_country = df_country[df_country['Travel Alert'].apply(alert_rank) <= max_rank]

    # === Visa exempt 濾器（修正版：支援數值/布林/中英文） ===
    def is_exempt(val):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return (not np.isnan(val)) and (val > 0)
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in {'1', 'true', 'yes', 'y', '是', '免簽', '免簽證', 'exempt', '免'}
    if 'exempt' in (visa_only or []):
        if 'Visa_exempt_entry' in df_country.columns:
            df_country = df_country[df_country['Visa_exempt_entry'].apply(is_exempt)]

    if df_country.empty:
        return html.Div("沒有符合條件的國家（被 Travel Alert / Visa 過濾掉）。", style={'color': 'white'}), [], []

    
    # === 聚合住宿成本到國家層級 ===
    agg = df_travel.groupby('Destination', as_index=False).agg(
        trips=('Destination', 'count'),
        median_daily_acc_cost=('acc_daily_cost', 'median'),
        mean_daily_acc_cost=('acc_daily_cost', 'mean'),
        median_trip_acc_cost=('acc_trip_cost', 'median'),
        mean_trip_acc_cost=('acc_trip_cost', 'mean')
    )

    # 合併
    out = df_country.merge(agg, left_on='Destination', right_on='Destination', how='inner')\
                    .rename(columns={'Destination': 'Country'})

    # 型別轉換
    for col in ['CPI','PCE','Safety Index']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # === 成本 × CPI 調整（自動把 CPI 轉成成本倍率）===
    # 取「全體（可用）CPI 的中位數」當基準；CPI 高 → 調整後成本更高
    if 'CPI' in out.columns:
        cpi_median = out['CPI'].dropna().median()
    else:
        cpi_median = np.nan
    def adjust_cost(row):
        base = row['median_daily_acc_cost']
        cpi  = row['CPI'] if 'CPI' in row and pd.notna(row['CPI']) else np.nan
        if pd.notna(base) and pd.notna(cpi) and pd.notna(cpi_median) and cpi_median > 0:
            return base * (cpi / cpi_median)
        return base
    out['adj_daily_acc_cost'] = out.apply(adjust_cost, axis=1)

    # === 計分（0–100）：Safety↑、Adjusted Cost↓（只有 Safety/Cost 兩權重） ===
    def minmax(series):
        s = series.dropna()
        if s.empty:
            return pd.Series([np.nan]*len(series), index=series.index)
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series([0.5]*len(series), index=series.index)
        return (series - lo) / (hi - lo)

    s_safety = minmax(out['Safety Index']) if 'Safety Index' in out.columns else None
    s_cost_raw = minmax(out['adj_daily_acc_cost'])  # 越低越好
    s_cost = 1 - s_cost_raw if s_cost_raw is not None else None

    # 權重正規化（Safety / Cost）
    ws = (w_safety or 0); wc = (w_cost or 0)
    denom = (ws + wc) or 1
    ws, wc = ws/denom, wc/denom

    scores = []
    for i in range(len(out)):
        parts, wts = [], []
        if s_safety is not None and pd.notna(s_safety.iloc[i]): parts.append(s_safety.iloc[i]); wts.append(ws)
        if s_cost   is not None and pd.notna(s_cost.iloc[i]):   parts.append(s_cost.iloc[i]);   wts.append(wc)
        if len(parts) == 0 or sum(wts) == 0:
            scores.append(np.nan)
        else:
            # 若某一方權重為 0，等同於忽略該維度
            norm = sum(wts)
            wts = [w/norm for w in wts]
            scores.append(100 * sum(p*w for p, w in zip(parts, wts)))
    out['Score'] = scores

    # 排序：Score → Safety → 成本（調整後）
    out = out.sort_values(by=['Score','Safety Index','adj_daily_acc_cost'], ascending=[False, False, True])

    compare_countries = out['Country'].head(MAX_COMPARE_COUNTRIES).tolist()
    country_options = [{'label': c, 'value': c} for c in out['Country'].tolist()]

    # 欄位與顯示格式
    shown_cols = [
        'Country', 'Score', 'Safety Index', 'Travel Alert', 'CPI', 'PCE', 'Visa_exempt_entry',
        'trips', 'median_daily_acc_cost', 'adj_daily_acc_cost', 'median_trip_acc_cost'
    ]
    available_cols = [c for c in shown_cols if c in out.columns]
    out = out[available_cols]

    def fmt(x, nd=0):
        try:
            return None if pd.isna(x) else (f"{x:.{nd}f}")
        except Exception:
            return x

    out_display = out.copy()
    if 'Score' in out_display: out_display['Score'] = out_display['Score'].apply(lambda v: fmt(v, 0))
    for c in ['median_daily_acc_cost', 'adj_daily_acc_cost', 'median_trip_acc_cost']:
        if c in out_display: out_display[c] = out_display[c].apply(lambda v: fmt(v, 0))

    table_component = dash_table.DataTable(
        data=out_display.to_dict('records'),
        page_size=10,
        export_format='csv',
        sort_action='native',
        filter_action='native',
        style_data={'backgroundColor': '#deb522', 'color': 'black'},
        style_header={'backgroundColor': 'black', 'color': '#deb522', 'fontWeight': 'bold'},
        style_table={'overflowX': 'auto'},
        columns=[
            {'name': 'Country', 'id': 'Country'},
            {'name': 'Score', 'id': 'Score'},
            {'name': 'Safety Index', 'id': 'Safety Index'},
            {'name': 'Travel Alert', 'id': 'Travel Alert'},
            {'name': 'CPI', 'id': 'CPI'},
            {'name': 'PCE', 'id': 'PCE'},
            {'name': 'Visa_exempt_entry', 'id': 'Visa_exempt_entry'},
            {'name': 'Trips', 'id': 'trips'},
            {'name': 'Median Daily Acc Cost', 'id': 'median_daily_acc_cost'},
            {'name': 'Adj Daily Acc Cost (CPI)', 'id': 'adj_daily_acc_cost'},
            {'name': 'Median Trip Acc Cost', 'id': 'median_trip_acc_cost'},
        ]
    )

    return table_component, country_options, compare_countries


@app.callback(
    Output('planner-selected-countries', 'data'),
    [Input('planner-compare-country-selector', 'value'),
     Input('graph-tabs', 'value')]
)
def sync_planner_selected_countries(selected_countries, tab):
    if tab != 'planner':
        return no_update

    clean_list = []
    for country in selected_countries or []:
        if isinstance(country, str) and country not in clean_list:
            clean_list.append(country)
        if len(clean_list) >= MAX_COMPARE_COUNTRIES:
            break

    return clean_list


@app.callback(
    [Output('planner-compare-radar', 'children'),
     Output('planner-compare-bar', 'children'),
     Output('planner-compare-line', 'children'),
     Output('planner-compare-table-container', 'children')],
    [Input('planner-selected-countries', 'data'),
     Input('planner-compare-metrics', 'value'),
     Input('graph-tabs', 'value')]
)
def update_trip_planner_comparison(countries, metrics, tab):
    if tab != 'planner':
        return no_update, no_update, no_update, no_update

    metrics = metrics or []
    if not countries:
        msg = html.Div('請先透過上方條件找到至少一個國家。', style={'color': 'white'})
        return msg, msg, msg, msg

    if not metrics:
        msg = html.Div('請至少選擇一個比較指標。', style={'color': 'white'})
        return msg, msg, msg, msg

    df_result, limited_countries = prepare_country_compare_data(countries, metrics)
    if df_result.empty or not limited_countries:
        msg = html.Div('所選國家沒有足夠的比較數據。', style={'color': 'white'})
        return msg, msg, msg, msg

    radar_fig = build_compare_figure(df_result, 'radar', 'Trip Planner 雷達圖')
    bar_fig = build_compare_figure(df_result, 'bar', 'Trip Planner 長條圖')
    line_fig = build_compare_figure(df_result, 'line', 'Trip Planner 折線圖')

    radar_div = html.Div([dcc.Graph(figure=radar_fig)])
    bar_div = html.Div([dcc.Graph(figure=bar_fig)])
    line_div = html.Div([dcc.Graph(figure=line_fig)])

    summary_note = html.Small(
        f"目前比較國家：{', '.join(limited_countries)}",
        style={'color': '#deb522'}
    )
    table_component = create_compare_table_component(df_result)
    table_div = html.Div([summary_note, table_component])

    return radar_div, bar_div, line_div, table_div

if __name__ == '__main__':
    app.run(debug=False)
