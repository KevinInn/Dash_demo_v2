from dash import Dash, html, dcc, Input, Output, dash_table, callback, no_update
import dash_bootstrap_components as dbc
import pandas as pd

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
    if tab == 'travel':
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
        # 返回 'Attractions' 頁面的佈局
        return html.Div([
            dcc.Dropdown(
                options=[{'label': country, 'value': country} for country in country_list],
                value=['Australia'],
                id='attractions-dropdown',
                multi=True,  # 啟用多選功能
                style={
                    'backgroundColor': '#deb522',  # 下拉式選單背景顏色
                    'color': 'black'               # 下拉式選單文字顏色
                }
            ),
            html.Div(
                id='attractions-output-container',
                style = {'overflow-x': 'auto'}
            )
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
    Output('tabs-content-2', 'children'), # callback function output: id為'tabs-content-2'的 children（第172行程式碼） TODO: 改行數
    [Input('dropdown-pie-1', 'value'), # callback function input 1: id為'dropdown-pie-1'的 value值（第145行程式碼）
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
    Output('attractions-output-container', 'children'),
    Input('attractions-dropdown', 'value'),
    Input('graph-tabs', 'value')
)
def update_attractions_output(chosen_countries, tab):
    if tab != 'attractions':
        return no_update
    # 檢查是否有選擇國家
    if not chosen_countries:
        return html.Div("請選擇至少一個國家。", style={'color': 'white'})
    # 過濾出選擇的國家
    chosen_df = attractions_df[attractions_df['country'].isin(chosen_countries)]
    return dash_table.DataTable(
        data=chosen_df.to_dict('records'),
        page_size=10,
        style_data={
            'backgroundColor': '#deb522',
            'color': 'black',
        },
        style_header={
            'backgroundColor': 'black',  # 修改表頭背景顏色
            'color': '#deb522',          # 修改表頭文字顏色
            'fontWeight': 'bold',
        }
    )

if __name__ == '__main__':
    app.run(debug=False)
