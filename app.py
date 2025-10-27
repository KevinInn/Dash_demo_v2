# Import 所有相關套件
from dash import Dash, html, dcc, Input, State, Output, dash_table, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash_leaflet as dl
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 從./src導入所有自定義函數
from src.const import get_constants, ALERT_RANK_MAP, ALL_COMPARE_METRICS
from src.generate_visualization import generate_bar, generate_pie, generate_map, generate_box
from src.data_clean import travel_data_clean, countryinfo_data_clean, data_merge
from src.utils import (
    prepare_country_compare_data, build_compare_figure, generate_stats_card, is_exempt, minmax, fmt
)

#################
#### 基礎設置 ####
#################
# 加載欲分析的資料集
travel_df = pd.read_csv('./data/Travel_dataset.csv')  # 旅遊資訊
country_info_df = pd.read_csv('./data/country_info.csv')  # 國家資訊
attractions_df = pd.read_csv('./data/Attractions.csv')  # 景點資訊

# 進行資料前處理
travel_df = travel_data_clean(travel_df)
country_info_df = countryinfo_data_clean(country_info_df)

# 合併 travel_df 和 country_info_df，方便後續分析
df_merged = data_merge(travel_df, country_info_df)

# 設定 Overview 頁面預設值
_conts = [c for c in df_merged['Continent'].dropna().unique().tolist() if str(c).strip() != ""]
_dests = [d for d in df_merged['Destination'].dropna().unique().tolist() if str(d).strip() != ""]
_first_geo = (_conts[0] if _conts else (_dests[0] if _dests else None))
DEFAULTS = {
    "bar1_geo": _first_geo,
    "pie1_geo": _first_geo,
    "pie2_field": "Traveler nationality",
    "map1_geo": None,                 # None 代表 All
    "map2_metric": "Safety Index",
    "box1_geo": _first_geo,
    "box2_metric": "Accommodation cost",
}

# 獲取國家名稱列表（景點頁使用）
country_list = list(attractions_df['country'].unique())

# 切換頁面（如有需要可以自行增加）
def load_data(tab):
    if tab in ('travel', 'planner'):
        return df_merged

# 呼叫 ./src/const.py 中的 get_constants() 函式（畫面上方四格統計）
num_of_country, num_of_traveler, num_of_nationality, avg_days = get_constants(travel_df)

# 初始化應用程式
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title='Travel Data Analysis Dashboard', suppress_callback_exceptions=True)
server = app.server

# 外觀設定
tab_style = {
    'idle': {
        'borderRadius': '10px','padding': '0px','marginInline': '5px','display':'flex',
        'alignItems':'center','justifyContent':'center','fontWeight': 'bold',
        'backgroundColor': '#deb522','border':'none'
    },
    'active': {
        'borderRadius': '10px','padding': '0px','marginInline': '5px','display':'flex',
        'alignItems':'center','justifyContent':'center','fontWeight': 'bold','border':'none',
        'textDecoration': 'underline','backgroundColor': '#deb522'
    }
}

# ===== 版面配置 =====
app.layout = html.Div([
    dbc.Container([
        # 頂部 Logo 與 分頁選項
        dbc.Row([
            dbc.Col(html.Img(src="./assets/logo.png", height=100), width=5, style={'marginTop': '15px'}),
            dbc.Col(
                dcc.Tabs(id='graph-tabs', value='overview', children=[
                    dcc.Tab(label='Overview', value='overview',
                            style=tab_style['idle'], selected_style=tab_style['active']),
                    dcc.Tab(label='Trip Planner', value='planner',
                            style=tab_style['idle'], selected_style=tab_style['active']),
                    dcc.Tab(label='Attractions', value='attractions',
                            style=tab_style['idle'], selected_style=tab_style['active']),
                ], style={'height':'50px'}),
                width=7, style={'alignSelf': 'center'}
            ),
        ]),

        # 四格統計
        dbc.Row([
            dbc.Col(generate_stats_card("Country", num_of_country, "./assets/earth.svg"), width=3),
            dbc.Col(generate_stats_card("Traveler", num_of_traveler, "./assets/user.svg"), width=3),
            dbc.Col(generate_stats_card("Nationality", num_of_nationality, "./assets/earth.svg"), width=3),
            dbc.Col(generate_stats_card("Average Days", avg_days, "./assets/calendar.svg"), width=3),
        ], style={'marginBlock': '10px'}),

        dbc.Row([
            dcc.Tabs(id='tabs', value='travel', children=[
                dcc.Tab(label='Travel Data', value='travel',
                        style={'border':'1px line white','backgroundColor':'black','color': '#deb522','fontWeight': 'bold'},
                        selected_style={'border':'1px solid white','backgroundColor':'black','color': '#deb522','fontWeight': 'bold','textDecoration': 'underline'}),
            ], style={'padding': '0px'})
        ]),

        # Overview頁面的圖表放置區
        html.Div(id='graph-content')
    ], style={'padding': '0px'})
], style={'backgroundColor': 'black', 'minHeight': '100vh'})

# ====== 頁面切換內容 ======
@app.callback(
    Output('graph-content', 'children'),
    [Input('graph-tabs', 'value')]
)
def render_tab_content(tab):
    if tab == 'overview':
        # 建立地理選項（洲 + 國家）
        geo_options = [{'label': i, 'value': i}
                       for i in pd.concat([df_merged['Continent'], df_merged['Destination']]).dropna().unique()]

        return html.Div([
            # 第一排：長條 + 圓餅
            dbc.Row([
                dbc.Col([
                    html.H3("各大洲或各國不同月份遊客人数", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-bar-1',
                        options=geo_options,
                        value=DEFAULTS["bar1_geo"],
                        placeholder='Select a continent or country',
                        style={'width': '90%','margin-top': '10px','margin-bottom': '10px'}
                    )
                ]),
                dbc.Col([
                    html.H3("各大洲或各國的遊客屬性、住宿及交通類型", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-pie-1',
                        options=geo_options,
                        value=DEFAULTS["pie1_geo"],
                        placeholder='Select a continent or country',
                        style={'width': '50%','margin':'5px 0','display': 'inline-block'}
                    ),
                    dcc.Dropdown(
                        id='dropdown-pie-2',
                        options=[{'label': i, 'value': i}
                                 for i in ['Traveler nationality','Age group','Traveler gender','Accommodation type','Transportation type']],
                        value=DEFAULTS["pie2_field"],
                        placeholder='Select a value',
                        style={'width': '50%','margin':'5px 0','display': 'inline-block'}
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([dcc.Loading([html.Div(id='tabs-content-1')], type='default', color='#deb522')]),
                dbc.Col([dcc.Loading([html.Div(id='tabs-content-2')], type='default', color='#deb522')]),
            ]),
            # 第二排：地圖 + 箱型圖
            dbc.Row([
                dbc.Col([
                    html.H3("各大洲或各國安全係數及消費水平", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-map-1',
                        options=[{'label': 'All', 'value': None}]
                                + [{'label': i, 'value': i} for i in df_merged['Continent'].dropna().unique()],
                        value=DEFAULTS["map1_geo"],
                        placeholder='Select a continent',
                        style={'width': '50%','margin':'5px 0','display': 'inline-block'}
                    ),
                    dcc.Dropdown(
                        id='dropdown-map-2',
                        options=[{'label': i, 'value': i} for i in ['Safety Index','Crime_index','CPI','PCE','Exchange_rate']],
                        value=DEFAULTS["map2_metric"],
                        placeholder='Select a value',
                        style={'width': '50%','margin':'5px 0','display': 'inline-block'}
                    )
                ]),
                dbc.Col([
                    html.H3("各大洲或各國家住宿及交通成本", style={'color': '#deb522', 'margin-top': '5px'}),
                    dcc.Dropdown(
                        id='dropdown-box-1',
                        options=[{'label': i, 'value': i}
                                 for i in pd.concat([df_merged['Continent'], df_merged['Destination']]).dropna().unique()],
                        value=DEFAULTS["box1_geo"],
                        placeholder='Select a continent or country',
                        style={'width': '50%','margin':'5px 0','display': 'inline-block'}
                    ),
                    dcc.Dropdown(
                        id='dropdown-box-2',
                        options=[{'label': i, 'value': i} for i in ['Accommodation cost','Transportation cost']],
                        value=DEFAULTS["box2_metric"],
                        placeholder='Select a value',
                        style={'width': '50%','margin':'5px 0','display': 'inline-block'}
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([dcc.Loading([html.Div(id='tabs-content-3')], type='default', color='#deb522')]),
                dbc.Col([dcc.Loading([html.Div(id='tabs-content-4')], type='default', color='#deb522')]),
            ]),
        ])

    elif tab == 'attractions':
        return html.Div([
            dcc.Dropdown(
                options=[{'label': country, 'value': country} for country in country_list],
                value='Australia', id='attractions-dropdown', multi=False,
                style={'backgroundColor': '#deb522', 'color': 'black'}
            ),
            html.Button(
                "查詢", id='attractions-submit', n_clicks=0, className="btn btn-primary",
                style={'backgroundColor': '#deb522','color': 'black','fontWeight': 'bold',
                       'marginTop': '10px','padding': '6px 16px','borderRadius': '6px','border': 'none','cursor': 'pointer'}
            ),
            dcc.Loading(
                id="attractions-loading", type="circle", color="#deb522", fullscreen=False,
                children=[html.Div(id='attractions-output-container', style={'overflow-x': 'auto','marginTop': '10px'}),
                          html.Div(id='attractions-map-container', style={'height': '600px','marginTop': '16px'})]
            )
        ])

    elif tab == 'planner':
        accommodation_types = sorted(travel_df['Accommodation type'].dropna().unique().tolist())

        # 動態取得所有 Travel Alert 值（聯集）
        alerts_from_country = country_info_df['Travel Alert'].dropna().astype(str).str.strip().tolist() \
                              if 'Travel Alert' in country_info_df.columns else []
        alerts_from_merged = df_merged['Travel Alert'].dropna().astype(str).str.strip().tolist() \
                             if 'Travel Alert' in df_merged.columns else []
        seen_alerts = sorted(set(alerts_from_country) | set(alerts_from_merged))

        default_rank = 3
        def alert_sort_key(c): return ALERT_RANK_MAP.get(c, default_rank)
        color_options = [{'label': c, 'value': c} for c in sorted(seen_alerts, key=alert_sort_key)]
        default_alert = ('灰色' if '灰色' in seen_alerts else (color_options[0]['value'] if color_options else None))

        return html.Div([
            dcc.Store(id='planner-selected-countries', data=[]),  # 只用來存「前五名」供比較圖表使用

            html.H3("Trip Planner：用預算、安全與住宿偏好找國家", style={'color': '#deb522', 'margin-top': '5px'}),

            # 篩選列 1：住宿費用 & 住宿類型
            dbc.Row([
                dbc.Col([
                    html.Label("Accommodation cost（min）", style={'color': '#deb522'}),
                    dcc.Input(id='planner-cost-min', type='number', placeholder='min',
                              style={'width': '100%','backgroundColor': 'black','color': '#deb522','border': '1px solid #deb522'})
                ], width=3),
                dbc.Col([
                    html.Label("Accommodation cost（max）", style={'color': '#deb522'}),
                    dcc.Input(id='planner-cost-max', type='number', placeholder='max',
                              style={'width': '100%','backgroundColor': 'black','color': '#deb522','border': '1px solid #deb522'})
                ], width=3),
                dbc.Col([
                    html.Label("Accommodation type（multi）", style={'color': '#deb522'}),
                    dcc.Dropdown(id='planner-acc-types',
                                 options=[{'label': t, 'value': t} for t in accommodation_types],
                                 value=[], multi=True, style={'backgroundColor': '#deb522','color': 'black'})
                ], width=6),
            ], style={'marginTop': '8px', 'marginBottom': '12px'}),

            # 篩選列 2：安全顏色門檻、免簽
            dbc.Row([
                dbc.Col([
                    html.Label("可接受的最高危險顏色（含以下等級）", style={'color': '#deb522'}),
                    dcc.Dropdown(
                        id='planner-alert-max', options=color_options, value=default_alert, clearable=False,
                        style={'backgroundColor': '#deb522', 'color': 'black'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Visa", style={'color': '#deb522'}),
                    dcc.Checklist(
                        id='planner-visa-only',
                        options=[{'label': ' 只顯示免簽', 'value': 'exempt'}],
                        value=[], inputStyle={'marginRight': '6px'}, labelStyle={'color': '#deb522'}
                    )
                ], width=6),
            ], style={'marginBottom': '12px'}),

            # 權重
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

            # 比較圖表
            html.H4("建議國家比較", style={'color': '#deb522', 'marginTop': '10px'}),
            dbc.Row([
                dbc.Col([dcc.Loading(html.Div(id='planner-compare-radar'), type='default', color='#deb522')], width=6),
                dbc.Col([dcc.Loading(html.Div(id='planner-compare-bar'), type='default', color='#deb522')], width=6),
            ], style={'marginBottom': '12px'}),
            dbc.Row([
                dbc.Col([dcc.Loading(html.Div(id='planner-compare-line'), type='default', color='#deb522')], width=12),
            ], style={'marginBottom': '12px'}),
        ])

    return html.Div("選擇的標籤頁不存在。", style={'color': 'white'})

####################################
#### Overview 頁面圖表 callbacks ####
####################################
@app.callback(
    Output('tabs-content-1', 'children'),
    [Input('dropdown-bar-1', 'value'), Input('graph-tabs', 'value')]
)
def update_bar_chart(dropdown_value, tab):
    if tab != 'overview':
        return no_update
    df = load_data('travel')
    # 安全 fallback：若使用者清空下拉，回到預設
    geo = dropdown_value or DEFAULTS["bar1_geo"]
    fig1 = generate_bar(df, geo)
    return html.Div([dcc.Graph(id='graph1', figure=fig1)], style={'width': '90%','display': 'inline-block'})

@app.callback(
    Output('tabs-content-2', 'children'),
    [Input('dropdown-pie-1', 'value'), Input('dropdown-pie-2', 'value'), Input('graph-tabs', 'value')]
)
def update_pie_chart(dropdown_value_1, dropdown_value_2, tab):
    if tab != 'overview':
        return no_update
    df = load_data('travel')
    geo = dropdown_value_1 or DEFAULTS["pie1_geo"]
    field = dropdown_value_2 or DEFAULTS["pie2_field"]
    fig2 = generate_pie(df, geo, field)
    return html.Div([dcc.Graph(id='graph2', figure=fig2)], style={'width': '90%','display': 'inline-block'})

@app.callback(
    Output('tabs-content-3', 'children'),
    [Input('dropdown-map-1', 'value'), Input('dropdown-map-2', 'value'), Input('graph-tabs', 'value')]
)
def update_map(dropdown_value_1, dropdown_value_2, tab):
    if tab != 'overview':
        return no_update
    df = load_data('travel')
    geo = dropdown_value_1 if dropdown_value_1 in _conts or dropdown_value_1 is None else DEFAULTS["map1_geo"]
    metric = dropdown_value_2 or DEFAULTS["map2_metric"]
    fig3 = generate_map(df, geo, metric)
    return html.Div([dcc.Graph(id='graph3', figure=fig3)], style={'width': '90%','display': 'inline-block'})

@app.callback(
    Output('tabs-content-4', 'children'),
    [Input('dropdown-box-1', 'value'), Input('dropdown-box-2', 'value'), Input('graph-tabs', 'value')]
)
def update_box_chart(dropdown_value_1, dropdown_value_2, tab):
    if tab != 'overview':
        return no_update
    df = load_data('travel')
    geo = dropdown_value_1 or DEFAULTS["box1_geo"]
    metric = dropdown_value_2 or DEFAULTS["box2_metric"]
    fig4 = generate_box(df, geo, metric)
    return html.Div([dcc.Graph(id='graph4', figure=fig4)], style={'width': '90%','display': 'inline-block'})

####################################
#### Trip Planner 頁面 callbacks ####
####################################
# 表格 
@app.callback(
    [Output('planner-table-container', 'children'),
     Output('planner-selected-countries', 'data')],
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
        return no_update, no_update

    df_travel = travel_df.copy()

    # 防呆：住宿成本 min/max 對調
    if cost_min is not None and cost_max is not None and cost_min > cost_max:
        cost_min, cost_max = cost_max, cost_min

    # 預處理
    df_travel['Accommodation cost'] = pd.to_numeric(df_travel['Accommodation cost'], errors='coerce').fillna(0)
    df_travel['Duration (days)'] = pd.to_numeric(df_travel['Duration (days)'], errors='coerce')
    df_travel = df_travel.dropna(subset=['Duration (days)'])
    df_travel = df_travel[df_travel['Duration (days)'] > 0]

    df_travel['acc_trip_cost']  = df_travel['Accommodation cost']
    df_travel['acc_daily_cost'] = df_travel['acc_trip_cost'] / df_travel['Duration (days)']

    # 條件 1：住宿成本區間
    if cost_min is not None:
        df_travel = df_travel[df_travel['Accommodation cost'] >= float(cost_min)]
    if cost_max is not None:
        df_travel = df_travel[df_travel['Accommodation cost'] <= float(cost_max)]

    # 條件 2：住宿類型多選
    if acc_types:
        df_travel = df_travel[df_travel['Accommodation type'].isin(acc_types)]

    if df_travel.empty:
        return html.Div("沒有符合條件的國家。", style={'color': 'white'}), []

    matched_countries = sorted(df_travel['Destination'].dropna().unique().tolist())

    # 取國家層欄位（避免取到全空）
    cols_needed = ['Destination', 'CPI', 'PCE', 'Safety Index', 'Visa_exempt_entry', 'Travel Alert']
    avail_cols = [c for c in cols_needed if c in df_merged.columns]
    sub = df_merged[df_merged['Destination'].isin(matched_countries)][avail_cols].copy()

    def first_nonnull(series):
        for v in series:
            if pd.notna(v) and str(v).strip() != '':
                return v
        return np.nan

    agg_spec = {col: first_nonnull for col in avail_cols if col != 'Destination'}
    df_country = sub.groupby('Destination', as_index=False).agg(agg_spec)
    key_cols = ['CPI', 'PCE', 'Safety Index', 'Travel Alert']
    df_country = df_country.dropna(subset=key_cols, how='any')

    # Travel Alert 門檻
    default_rank = 3
    def alert_rank(v): return ALERT_RANK_MAP.get(str(v).strip(), default_rank)
    if alert_max is not None:
        max_rank = alert_rank(alert_max)
        df_country = df_country[df_country['Travel Alert'].apply(alert_rank) <= max_rank]

    if 'exempt' in (visa_only or []):
        if 'Visa_exempt_entry' in df_country.columns:
            df_country = df_country[df_country['Visa_exempt_entry'].apply(is_exempt)]

    if df_country.empty:
        return html.Div("沒有符合條件的國家（被 Travel Alert / Visa 過濾掉）。", style={'color': 'white'}), []

    # 聚合住宿成本到國家層級
    agg = df_travel.groupby('Destination', as_index=False).agg(
        trips=('Destination', 'count'),
        median_daily_acc_cost=('acc_daily_cost', 'median'),
        mean_daily_acc_cost=('acc_daily_cost', 'mean'),
        median_trip_acc_cost=('acc_trip_cost', 'median'),
        mean_trip_acc_cost=('acc_trip_cost', 'mean')
    )

    out = df_country.merge(agg, on='Destination', how='inner').rename(columns={'Destination': 'Country'})

    for col in ['CPI','PCE','Safety Index']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')

    # 成本 × CPI 調整
    cpi_median = out['CPI'].dropna().median() if 'CPI' in out.columns else np.nan
    def adjust_cost(row):
        base = row['median_daily_acc_cost']
        cpi  = row['CPI'] if 'CPI' in row and pd.notna(row['CPI']) else np.nan
        if pd.notna(base) and pd.notna(cpi) and pd.notna(cpi_median) and cpi_median > 0:
            return base * (cpi / cpi_median)
        return base
    out['adj_daily_acc_cost'] = out.apply(adjust_cost, axis=1)

    s_safety = minmax(out['Safety Index']) if 'Safety Index' in out.columns else None
    s_cost_raw = minmax(out['adj_daily_acc_cost'])
    s_cost = 1 - s_cost_raw if s_cost_raw is not None else None

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
            norm = sum(wts); wts = [w/norm for w in wts]
            scores.append(100 * sum(p*w for p, w in zip(parts, wts)))
    out['Score'] = scores

    # 排序 + 取前五
    out = out.sort_values(by=['Score','Safety Index','adj_daily_acc_cost'], ascending=[False, False, True])
    compare_countries = out['Country'].head(5).tolist()

    # 顯示表格（簡要）
    shown_cols = [
        'Country','Score','Safety Index','Travel Alert','CPI','PCE','Visa_exempt_entry',
        'trips','median_daily_acc_cost','adj_daily_acc_cost','median_trip_acc_cost'
    ]
    available_cols = [c for c in shown_cols if c in out.columns]
    out_display = out[available_cols].copy()

    if 'Score' in out_display: out_display['Score'] = out_display['Score'].apply(lambda v: fmt(v, 0))
    for c in ['median_daily_acc_cost','adj_daily_acc_cost','median_trip_acc_cost']:
        if c in out_display: out_display[c] = out_display[c].apply(lambda v: fmt(v, 0))

    table_component = dash_table.DataTable(
        data=out_display.to_dict('records'), page_size=10, export_format='csv',
        sort_action='native', filter_action='native',
        style_data={'backgroundColor': '#deb522', 'color': 'black'},
        style_header={'backgroundColor': 'black', 'color': '#deb522', 'fontWeight': 'bold'},
        style_table={'overflowX': 'auto'},
        columns=[{'name': col, 'id': col} for col in available_cols]
    )

    return table_component, compare_countries

# 產生雷達 / 長條 / 折線圖（前五名 + 全指標）
@app.callback(
    [Output('planner-compare-radar', 'children'),
     Output('planner-compare-bar', 'children'),
     Output('planner-compare-line', 'children')],
    [Input('planner-selected-countries', 'data'),
     Input('graph-tabs', 'value')]
)
def update_trip_planner_comparison(countries, tab):
    if tab != 'planner':
        return no_update, no_update, no_update

    if not countries:
        msg = html.Div('請先透過上方條件找到至少一個國家。', style={'color': 'white'})
        return msg, msg, msg
    metrics = ALL_COMPARE_METRICS  # 預設比較所有指標
    df_result, limited_countries = prepare_country_compare_data(countries, metrics, df_merged)
    if df_result.empty or not limited_countries:
        msg = html.Div('所選國家沒有足夠的比較數據。', style={'color': 'white'})
        return msg, msg, msg

    radar_fig = build_compare_figure(df_result, 'radar', 'Trip Planner 雷達圖')
    bar_fig = build_compare_figure(df_result, 'bar', 'Trip Planner 長條圖')
    line_fig = build_compare_figure(df_result, 'line', 'Trip Planner 折線圖')

    return html.Div([dcc.Graph(figure=radar_fig)]), \
           html.Div([dcc.Graph(figure=bar_fig)]), \
           html.Div([dcc.Graph(figure=line_fig)])

###############################
#### Attractions callback ####
###############################
@app.callback(
    [Output('attractions-output-container', 'children'),
     Output('attractions-map-container', 'children')],
    [Input('attractions-submit', 'n_clicks'), Input('graph-tabs', 'value')],
    [State('attractions-dropdown', 'value')],
    prevent_initial_call=True
)
def update_attractions_output(n_clicks, tab, chosen_country):
    if tab != 'attractions':
        raise PreventUpdate
    if n_clicks == 0 or not chosen_country:
        return (html.Div("請選擇一個國家並按下查詢。", style={'color': 'white'}), no_update)

    chosen_df = attractions_df[attractions_df['country'] == chosen_country].copy()

    table = dash_table.DataTable(
        data=chosen_df.to_dict('records'), page_size=10,
        style_data={'backgroundColor': '#deb522', 'color': 'black'},
        style_header={'backgroundColor': 'black', 'color': '#deb522', 'fontWeight': 'bold'}
    )

    def pick_col(cands):
        for c in cands:
            if c in chosen_df.columns:
                return c
        return None
    name_col = pick_col(['attraction', 'Attraction']) or chosen_df.columns[0]

    geolocator = Nominatim(user_agent="my_dash_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    points = []
    for _, r in chosen_df.iterrows():
        name = str(r[name_col])
        try:
            location = geocode(name)
            if location:
                points.append({'name': name, 'lat': location.latitude, 'lng': location.longitude})
        except Exception:
            continue

    if not points:
        return table, html.Div("選定國家目前沒有可用座標的景點。", style={'color': 'white'})

    tile_layer = dl.TileLayer(
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )
    markers = [dl.Marker(position=[p['lat'], p['lng']], children=dl.Tooltip(p['name'])) for p in points]

    lats = [p['lat'] for p in points]; lngs = [p['lng'] for p in points]
    south, west = min(lats), min(lngs); north, east = max(lats), max(lngs)
    bounds = [[south, west], [north, east]]

    if len(points) == 1:
        center = [points[0]['lat'], points[0]['lng']]
        the_map = dl.Map(id=f"map-{hash(str(bounds))}",
                         children=[tile_layer, dl.LayerGroup(markers)],
                         center=center, zoom=10, style={'width': '100%','height': '600px'})
    else:
        the_map = dl.Map(id=f"map-{hash(str(bounds))}",
                         children=[tile_layer, dl.LayerGroup(markers)],
                         bounds=bounds, style={'width': '100%','height': '600px'})
    return table, the_map

if __name__ == '__main__':
    app.run(debug=False)
