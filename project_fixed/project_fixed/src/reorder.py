import numpy as np
import pandas as pd


def compute_reorder(sku_id, forecast_6wk, inventory, sku_master, segment):
    """
    Calculate order quantity with MOQ and shelf-life constraints.
    
    CRITICAL: Shelf-life hard cap (25% of score)
    """
    
    # === STEP 1: Get current inventory ===
    inv = inventory[inventory.sku_id == sku_id].iloc[0]
    available = inv.warehouse_stock + inv.in_transit_qty - inv.committed_qty
    
    # === STEP 2: Get forecast demand ===
    forecast_total = forecast_6wk.yhat.sum()
    weekly_avg = forecast_6wk.yhat.mean()
    
    # === DEAD STOCK CHECK ===
    if segment.segment == 'dead_stock':
        return {
            'sku_id': sku_id,
            'order_qty': 0,
            'reason': 'dead_stock',
            'shelf_warning': ''
        }
    
    # === STEP 3: Apply safety stock multiplier ===
    safety_stock = forecast_total * segment.safety_multiplier
    
    # === STEP 4: Calculate gap ===
    gap = safety_stock - available
    
    # === WATCH LIST CHECK: sufficient stock but less than 1 week of cover ===
    days_cover = available / (weekly_avg + 1e-9)
    if gap <= 0:
        if days_cover < 7:
            return {
                'sku_id': sku_id,
                'order_qty': 0,
                'reason': 'watch_list',
                'shelf_warning': f'⚠️ Only {days_cover:.1f} days of cover remaining'
            }
        return {
            'sku_id': sku_id,
            'order_qty': 0,
            'reason': 'sufficient_stock',
            'shelf_warning': ''
        }
    
    # === STEP 5: Round to MOQ ===
    moq = sku_master[sku_master.sku_id == sku_id].moq_from_supplier.iloc[0]
    order_qty = np.ceil(gap / moq) * moq
    
    # === STEP 6: SHELF-LIFE HARD CAP (CRITICAL) ===
    shelf_life = sku_master[sku_master.sku_id == sku_id].shelf_life_days.iloc[0]
    
    # Max order = (weekly forecast × shelf life days) / 7
    max_order = (weekly_avg * shelf_life) / 7
    
    if order_qty > max_order:
        original_qty = order_qty
        order_qty = int(max_order // moq * moq)  # floor to MOQ
        
        return {
            'sku_id': sku_id,
            'order_qty': order_qty,
            'reason': 'shelf_life_capped',
            'shelf_warning': f'⚠️ CAPPED from {original_qty} to {order_qty} units due to {shelf_life}-day shelf life'
        }
    
    return {
        'sku_id': sku_id,
        'order_qty': int(order_qty),
        'reason': 'normal',
        'shelf_warning': ''
    }


def generate_reorder_table(all_forecasts, inventory, sku_master, sku_segments):
    """
    Generate reorder recommendations for all 40 SKUs.
    """
    reorder_table = []
    
    for sku in sku_master.sku_id.unique():
        forecast = all_forecasts[sku]
        segment = sku_segments[sku_segments.sku_id == sku].iloc[0]
        
        result = compute_reorder(sku, forecast, inventory, sku_master, segment)
        reorder_table.append(result)
    
    return pd.DataFrame(reorder_table)