import plotly.express as px
import plotly.colors as colors
import pandas as pd

# 視覺化統計圖

# 長條圖
def generate_bar(df, dropdown_value):
    if dropdown_value is None:
        # 回傳一個空的圖表，或在這裡設置一個預設訊息
        fig_bar = px.bar(title="請選擇有效的選項")
        fig_bar.update_layout(template='plotly_dark', font=dict(color='#deb522'))
    
        return fig_bar

    # 長條圖的x軸要依照月份順序排列
    # 定義月份的順序
    month_order = ['January', 'February', 'March', 'April', 'May', 
                'June', 'July', 'August', 'September', 'October', 
                'November', 'December']

    # 過濾資料
    df_group = df[(df['Continent'] == dropdown_value) | (df['Destination'] == dropdown_value)]

    # 計算 'Start month' 的數量
    month_counts = df_group['Start month'].value_counts().reindex(month_order, fill_value=0).reset_index()
    month_counts.columns = ['Start month', 'count']  # 設定新列名

    # 計算百分比
    month_counts['percentage'] = (month_counts['count'] / month_counts['count'].sum()) * 100

    # 將 'Start month' 列轉換為類別型資料並設置順序
    month_counts['Start month'] = pd.Categorical(month_counts['Start month'], categories=month_order, ordered=True)
        
    # 依照月份順序排序
    month_counts = month_counts.sort_values(by='Start month')

    fig_bar = px.bar(month_counts, x='Start month', y='count',
                    color='count', text='percentage',
                    title=f'{dropdown_value} - Traveler number each month',
                    labels={'count': 'Count', 'index': 'Start month', 'percentage': 'Percentage'},
                    color_continuous_scale='Viridis')
    
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_bar.update_layout(yaxis=dict(categoryorder='total ascending'))
    fig_bar.update_layout(template='plotly_dark', font=dict(color='#deb522'))

    return fig_bar

def generate_pie(df, dropdown_value_1, dropdown_value_2):

    if dropdown_value_1 is None or dropdown_value_2 is None:
        # 回傳一個空的圖表，或在這裡設置一個預設訊息
        fig_pie = px.pie(title="請選擇有效的選項")
        fig_pie.update_layout(template='plotly_dark', font=dict(color='#deb522'))
    
        return fig_pie
 
    # 過濾出符合 `dropdown_value_1` 的資料
    df_group = df[(df['Continent'] == dropdown_value_1) | (df['Destination'] == dropdown_value_1)]
    
    # 使用 `value_counts()` 計算 `dropdown_value_2` 欄位的次數，並重置索引以創建新的資料框
    df_counts = df_group[dropdown_value_2].value_counts().reset_index(name = 'count')
    
    # 建立圓餅圖，使用 `dropdown_value_2` 作為標籤，`count` 作為數值
    fig_pie = px.pie(
        df_counts, 
        names=dropdown_value_2,  # 圓餅圖的標籤欄位
        values='count',  # 圓餅圖的數值欄位
        title=f'{dropdown_value_1} - {dropdown_value_2}',
        color_discrete_sequence=colors.sequential.Viridis  # 圖表標題
    )
        
    # 更新圖表樣式
    fig_pie.update_layout(template='plotly_dark', font=dict(color='#deb522'))

    return fig_pie

def generate_map(df, dropdown_value_1, dropdown_value_2):

    if dropdown_value_1 is None and dropdown_value_2 is None:
        # 回傳一個空的圖表，或在這裡設置一個預設訊息
        fig_choropleth = px.choropleth(title="請選擇有效的選項")
        fig_choropleth.update_layout(template='plotly_dark', font=dict(color='#deb522'))
    
        return fig_choropleth

    country_mapping = {
        'USA': 'USA',
        'UK': 'GB-ENG',
        'France': 'FRA',
        'Canada': 'CAN',
        'Germany': 'DEU',
        'Japan': 'JPN',
        'Australia': 'AUS',
        'Italy': 'ITA',
        'Spain': 'ESP',
        'Mexico': 'MEX',
        'New Zealand': 'NZL',
        'South Korea': 'KOR',
        'United Arab Emirates': 'ARE',
        'Netherlands': 'NLD',
        'South Africa': 'ZAF',
        'Thailand': 'THA',
        'Egypt': 'EGY',
        'Brazil': 'BRA',
        'Morocco': 'MAR',
        'Indonesia': 'IDN',
        'Scotland': 'GB-SCT',
        'Greek': 'GRC',
        'Cambodia': 'KHM',  
    }

    if dropdown_value_1 != None:
        df_group = df[df['Continent'] == dropdown_value_1]  
    else:
        df_group = df.copy()

    df_group['Destination'] = df_group['Destination'].map(country_mapping)

    fig_choropleth = px.choropleth(df_group, 
                                    locations="Destination",
                                    color=dropdown_value_2,
                                    hover_name="Destination",
                                    title=f"{dropdown_value_1} - {dropdown_value_2}",
                                    projection="natural earth",
                                    color_continuous_scale='Viridis')
    
    fig_choropleth.update_layout(template='plotly_dark', font=dict(color='#deb522'))
    
    return fig_choropleth

def generate_box(df, dropdown_value_1, dropdown_value_2):

    if dropdown_value_1 is None or dropdown_value_2 is None:
        # 回傳一個空的圖表，或在這裡設置一個預設訊息
        fig_boxplot = px.box(title="請選擇有效的選項")
        fig_boxplot.update_layout(template='plotly_dark', font=dict(color='#deb522'))
    
        return fig_boxplot

    df_group = df[(df['Continent'] == dropdown_value_1) | (df['Destination'] == dropdown_value_1)]
    
    fig_boxplot = px.box(df_group, x=dropdown_value_2, title=f'{dropdown_value_1} - {dropdown_value_2}')
    fig_boxplot.update_traces(marker=dict(color='#deb522'))
    fig_boxplot.update_layout(template='plotly_dark', font=dict(color='#deb522'))

    return fig_boxplot