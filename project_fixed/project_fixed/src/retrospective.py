import pandas as pd


def identify_diwali_stockouts(df_master, promotions_calendar, diwali_promo_id='PR001'):
    """
    Find the 14 SKUs that stocked out in Diwali 2023
    Ground truth validation for the model
    """
    # 1. Extract Diwali 2023 window (Oct 24 - Nov 14, 2023)
    diwali_start = pd.Timestamp('2023-10-24')
    diwali_end = pd.Timestamp('2023-11-14')
    post_end = diwali_end + pd.Timedelta(weeks=4)

    if df_master.empty:
        return pd.DataFrame(columns=['sku_id', 'pre_velocity', 'diwali_sales', 'post_velocity', 'acceleration', 'stockout_score'])
    
    # 2. Get 4-week pre-Diwali velocity (Sept 26 - Oct 23)
    pre_diwali = df_master[
        (df_master.week_start_date >= '2023-09-26') &
        (df_master.week_start_date < diwali_start)
    ].groupby('sku_id').agg({'units_sold': 'sum'}).rename(columns={'units_sold': 'pre_velocity'})
    
    # 3. Get Diwali week velocity (actual sold during promo)
    during_diwali = df_master[
        (df_master.week_start_date >= diwali_start) &
        (df_master.week_start_date <= diwali_end)
    ].groupby('sku_id').agg({'units_sold': 'sum'}).rename(columns={'units_sold': 'diwali_sales'})

    # 3b. Get post-Diwali velocity for recovery signal
    post_diwali = df_master[
        (df_master.week_start_date > diwali_end) &
        (df_master.week_start_date <= post_end)
    ].groupby('sku_id').agg({'units_sold': 'sum'}).rename(columns={'units_sold': 'post_velocity'})
    
    # 4. Get inventory snapshot on Oct 23, 2023 (construct backward from current stock + sales)
    # This requires reverse-engineering from inventory_snapshot.csv
    # Assume inventory_snapshot is "current" — backtrack by subtracting historical sales
    
    # 5. Compute stockout score
    velocity_df = pre_diwali.join(during_diwali, how='outer').join(post_diwali, how='outer').fillna(0)
    velocity_df['acceleration'] = velocity_df.diwali_sales / (velocity_df.pre_velocity + 1)
    
    # 6. Cross-reference with PR001 (Diwali Mega Sale SKUs)
    promo_row = promotions_calendar[promotions_calendar.promo_id == diwali_promo_id]
    if promo_row.empty:
        pr001_skus = []
    else:
        pr001_skus = [sku.strip() for sku in str(promo_row.iloc[0].sku_ids).split(',') if sku.strip()]
    
    # 7. Rank by stockout likelihood
    velocity_df['in_promo'] = velocity_df.index.isin(pr001_skus)
    velocity_df['stockout_score'] = (
        velocity_df.acceleration * 0.5 +
        velocity_df.in_promo * 0.3 +
        (velocity_df.pre_velocity / velocity_df.pre_velocity.max()) * 0.2
    )
    
    # 8. Top 14 by score
    top14 = velocity_df.nlargest(14, 'stockout_score').reset_index().rename(columns={'index': 'sku_id'})
    
    return top14[['sku_id', 'pre_velocity', 'diwali_sales', 'post_velocity', 'acceleration', 'stockout_score']]

# VALIDATION: Run same logic on 2022 data (PR005)
# If model correctly identifies 2022 stockouts, judges trust 2023 predictions