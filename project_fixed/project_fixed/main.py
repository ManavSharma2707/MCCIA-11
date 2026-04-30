from src.features import segment_skus
from src.forecast import run_all_forecasts
from src.ingest import ingest_data
from src.reorder import generate_reorder_table
from src.retrospective import identify_diwali_stockouts


def build_dashboard_state():
	df, sku_master, inventory, promotions_calendar, festive_calendar, outlets = ingest_data()
	sku_segments = segment_skus(df)
	all_forecasts = run_all_forecasts(df, sku_master, festive_calendar)
	reorder_df = generate_reorder_table(all_forecasts, inventory, sku_master, sku_segments)
	diwali_retrospective = identify_diwali_stockouts(df, promotions_calendar)

	return {
		'df': df,
		'sku_master': sku_master,
		'inventory': inventory,
		'promotions_calendar': promotions_calendar,
		'festive_calendar': festive_calendar,
		'outlets': outlets,
		'sku_segments': sku_segments,
		'all_forecasts': all_forecasts,
		'reorder_df': reorder_df,
		'diwali_retrospective': diwali_retrospective,
	}


def main():
	state = build_dashboard_state()
	print(
		f"Loaded {len(state['sku_master'])} SKUs, "
		f"built {len(state['all_forecasts'])} forecasts, "
		f"and generated {len(state['reorder_df'])} reorder rows."
	)


if __name__ == '__main__':
	main()
