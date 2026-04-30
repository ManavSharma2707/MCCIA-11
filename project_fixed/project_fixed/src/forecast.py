from prophet import Prophet

def forecast_sku(sku_id, df, festive_cal, horizon_weeks=6):
    """
    Fit Prophet model per SKU with Diwali regressor.
    Returns 6-week forecast.
    """
    
    # Filter to single SKU, aggregate to weekly level
    sku_df = df[df.sku_id == sku_id].groupby('week_start_date').agg({
        'baseline_sales': 'sum',  # use baseline (promo-stripped)
        'promotional_flag': 'max'  # 1 if any promo that week
    }).reset_index()
    
    # Prophet requires: ds (date), y (target)
    sku_df = sku_df.rename(columns={'week_start_date': 'ds', 'baseline_sales': 'y'})
    
    # Add Diwali regressor from festive_calendar
    sku_df = sku_df.merge(
        festive_cal[['date', 'demand_impact']].rename(columns={'date': 'ds'}),
        on='ds',
        how='left'
    )
    sku_df['diwali_flag'] = (sku_df.demand_impact == 'Very High').fillna(False).astype(int)
    
    # Fit Prophet
    model = Prophet(
        yearly_seasonality=False,  # only 3 years of data
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_regressor('promotional_flag')
    model.add_regressor('diwali_flag')
    
    model.fit(sku_df[['ds', 'y', 'promotional_flag', 'diwali_flag']])
    
    # === PREDICT 6 WEEKS AHEAD ===
    future = model.make_future_dataframe(periods=horizon_weeks, freq='W')
    
    # Set future regressors
    # Assume no promotions in next 6 weeks (unless explicitly scheduled)
    future['promotional_flag'] = 0
    
    # Check if Diwali falls in next 6 weeks
    future = future.merge(
        festive_cal[['date', 'demand_impact']].rename(columns={'date': 'ds'}),
        on='ds',
        how='left'
    )
    future['diwali_flag'] = (future.demand_impact == 'Very High').fillna(False).astype(int)
    
    forecast = model.predict(future)
    
    # Return only the 6 future weeks
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon_weeks)


def run_all_forecasts(df, sku_master, festive_cal):
    """
    Loop over all 40 SKUs (NOT 140).
    Runtime: ~40 seconds (1 sec per SKU).
    """
    all_forecasts = {}
    
    for sku in sku_master.sku_id.unique():
        print(f"Forecasting {sku}...")
        all_forecasts[sku] = forecast_sku(sku, df, festive_cal)
    
    print(f"✅ Forecasted all {len(all_forecasts)} SKUs")
    
    return all_forecasts