import pandas as pd
import numpy as np

def prepare_country_compare_data(countries, metrics, df_merged):
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
    limited_countries = [c for c in deduped if c in available_destinations][:5]

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
