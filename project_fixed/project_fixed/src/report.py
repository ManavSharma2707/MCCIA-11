from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

def generate_monday_report(reorder_df, diwali_retrospective, sku_master):
    """
    Generate PDF report with 3 sections + Diwali validation.
    """
    
    # Merge with SKU details for display
    reorder_df = reorder_df.merge(
        sku_master[['sku_id', 'product_name', 'brand', 'unit_price', 'moq_from_supplier']].rename(
            columns={'brand': 'supplier', 'unit_price': 'unit_cost'}
        ),
        on='sku_id'
    )
    
    # === SECTION A: Order Immediately ===
    section_a = reorder_df[reorder_df.order_qty > 0].sort_values('order_qty', ascending=False)
    
    # === SECTION B: Watch List (near stockout) ===
    # SKUs with available stock < 1 week forecast
    section_b = reorder_df[
        (reorder_df.order_qty == 0) &
        (reorder_df.reason == 'watch_list')  # flagged by logic
    ]
    
    # === SECTION C: Dead Stock ===
    section_c = reorder_df[reorder_df.reason == 'dead_stock']
    
    # Load Jinja2 template
    base_dir = Path(__file__).resolve().parents[1]
    env = Environment(loader=FileSystemLoader(str(base_dir / 'templates')))
    template = env.get_template('monday_report.html')
    
    # Render HTML
    html_content = template.render(
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
        order_now=section_a.to_dict('records'),
        watch_list=section_b.to_dict('records'),
        dead_stock=section_c.to_dict('records'),
        diwali_retrospective=diwali_retrospective.to_dict('records')
    )
    
    # Convert to PDF
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'monday_report_{datetime.now().strftime("%Y%m%d")}.pdf'
    HTML(string=html_content).write_pdf(output_path)
    
    print(f"✅ Report generated: {output_path}")
    return str(output_path)


def start_report_scheduler(generate_report_callback, *args, **kwargs):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        generate_report_callback,
        'cron',
        day_of_week='mon',
        hour=7,
        minute=55,
        args=args,
        kwargs=kwargs,
    )
    scheduler.start()
    return scheduler