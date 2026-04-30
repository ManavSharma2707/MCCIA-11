import pandas as pd


def segment_skus(df):
    """
    Classify 40 SKUs into: fast_mover, slow_mover, seasonal, dead_stock
    Each segment gets different safety stock multiplier
    """
    
    # Compute velocity statistics per SKU
    sku_stats = df.groupby('sku_id').agg({
        'baseline_sales': ['mean', 'std', 'sum'],
        'week_start_date': 'count'
    }).reset_index()
    
    sku_stats.columns = ['sku_id', 'avg_weekly', 'std_weekly', 'total_sales', 'weeks_active']
    
    # Coefficient of Variation (volatility measure)
    # NOTE: column renamed from 'cov' to 'coeff_var' to avoid collision
    # with the built-in pandas Series.cov() method inside .apply()
    sku_stats['coeff_var'] = sku_stats.std_weekly / (sku_stats.avg_weekly + 1)
    
    # Percentiles
    p25 = sku_stats.avg_weekly.quantile(0.25)
    p75 = sku_stats.avg_weekly.quantile(0.75)
    
    # === CLASSIFICATION RULES ===
    
    def classify_segment(row):
        # Dead stock: no sales in last 8 weeks
        recent_sales = df[
            (df.sku_id == row.sku_id) &
            (df.week_start_date >= df.week_start_date.max() - pd.Timedelta(weeks=8))
        ].baseline_sales.sum()
        
        if recent_sales == 0:
            return 'dead_stock', 0.0  # no safety stock
        
        # Fast mover: high avg, low volatility
        if row.avg_weekly > p75 and row['coeff_var'] < 0.6:
            return 'fast_mover', 1.2
        
        # Slow mover: low avg, age > 16 weeks
        if row.avg_weekly < p25 and row.weeks_active > 16:
            return 'slow_mover', 0.8
        
        # Seasonal: high volatility + festive spikes
        if row['coeff_var'] > 0.6:
            # Check if SKU has spikes during festive weeks
            # (This requires joining with festive_calendar — simplified here)
            return 'seasonal', 1.5
        
        # Default: normal
        return 'normal', 1.0
    
    sku_stats[['segment', 'safety_multiplier']] = sku_stats.apply(
        classify_segment, axis=1, result_type='expand'
    )
    
    return sku_stats[['sku_id', 'segment', 'safety_multiplier']]