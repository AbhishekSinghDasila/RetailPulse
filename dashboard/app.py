"""
RetailPulse — Phase 5: Streamlit Dashboard
===========================================
Interactive executive analytics dashboard.

Usage:
    pip install streamlit plotly scikit-learn
    streamlit run dashboard/app.py

Then open http://localhost:8501
"""

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title="RetailPulse Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "retailpulse.db")
PALETTE = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626",
           "#0891B2", "#BE185D", "#65A30D"]
SNAPSHOT = "2024-06-30"

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #F8FAFC; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1E293B;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] h2 { color: #F1F5F9 !important; font-weight: 600; }

/* Section headers */
h1 { color: #0F172A !important; font-weight: 700 !important; }
h2 { color: #1E293B !important; font-weight: 600 !important; }
h3 { color: #334155 !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    padding: 8px 20px;
    font-weight: 500;
    color: #0F172A !important;
}
.stTabs [aria-selected="true"] {
    background: #2563EB !important; color: white !important;
    border-color: #2563EB !important;
}

/* Metric labels */
[data-testid="metric-container"] label,
[data-testid="metric-container"] p,
[data-testid="metric-container"] div {
    color: #0F172A !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* Metric delta */
[data-testid="stMetricDelta"] {
    color: #16A34A !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATA LAYER — cached queries
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_all():
    conn = sqlite3.connect(DB_PATH)
    customers   = pd.read_sql("SELECT * FROM customers",   conn, parse_dates=["signup_date"])
    products    = pd.read_sql("SELECT * FROM products",    conn)
    orders      = pd.read_sql("SELECT * FROM orders",      conn, parse_dates=["order_date"])
    order_items = pd.read_sql("SELECT * FROM order_items", conn)
    sessions    = pd.read_sql("SELECT * FROM sessions",    conn, parse_dates=["session_start"])
    conn.close()
    return customers, products, orders, order_items, sessions


@st.cache_data(ttl=300)
def build_rfm():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT
            o.customer_id,
            c.segment, c.city, c.acquisition_channel, c.age, c.gender,
            CAST(julianday('{SNAPSHOT}') - julianday(MAX(o.order_date)) AS INTEGER) AS recency_days,
            COUNT(DISTINCT o.order_id)   AS frequency,
            ROUND(SUM(o.total_amount),2) AS monetary,
            ROUND(AVG(o.total_amount),2) AS avg_order_value
        FROM orders o JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.status = 'Delivered'
        GROUP BY o.customer_id
    """, conn)
    conn.close()
    df["r"] = pd.qcut(df["recency_days"], 5, labels=[5,4,3,2,1]).astype(int)
    df["f"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    df["m"] = pd.qcut(df["monetary"],    5, labels=[1,2,3,4,5]).astype(int)
    df["rfm_score"] = df["r"] + df["f"] + df["m"]
    df["rfm_segment"] = pd.cut(df["rfm_score"],
        bins=[0,6,9,12,15], labels=["Hibernating","At-Risk","Loyal","Champions"])
    return df


@st.cache_data(ttl=300)
def monthly_revenue():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT strftime('%Y-%m', order_date) AS month,
               COUNT(*) AS orders,
               ROUND(SUM(total_amount)/1e6,2) AS revenue_m,
               ROUND(SUM(CASE WHEN status='Delivered' THEN total_amount END)/1e6,2) AS del_rev_m
        FROM orders
        GROUP BY month ORDER BY month
    """, conn)
    conn.close()
    df["month"] = pd.to_datetime(df["month"])
    df["mom_growth"] = df["revenue_m"].pct_change() * 100
    return df


@st.cache_data(ttl=300)
def category_revenue():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT p.category,
               ROUND(SUM(oi.line_total)/1e6,2) AS revenue_m,
               COUNT(DISTINCT o.order_id)       AS orders,
               ROUND(AVG(p.price),0)            AS avg_price
        FROM order_items oi
        JOIN orders   o ON oi.order_id   = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.status = 'Delivered'
        GROUP BY p.category ORDER BY revenue_m DESC
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def funnel_data():
    conn = sqlite3.connect(DB_PATH)
    sessions = pd.read_sql("SELECT pages_viewed, converted, device, channel FROM sessions", conn)
    conn.close()
    pages = ["Home","Category","Product","Cart","Checkout","Confirmation"]
    rows = []
    for depth, page in enumerate(pages, 1):
        rows.append({
            "stage": page,
            "reached": (sessions["pages_viewed"] >= depth).sum(),
        })
    df = pd.DataFrame(rows)
    df["drop_pct"] = (1 - df["reached"] / df["reached"].shift(1)) * 100
    return df, sessions


@st.cache_data(ttl=300)
def forecast_data():
    conn = sqlite3.connect(DB_PATH)
    monthly = pd.read_sql("""
        SELECT DATE(order_date, 'start of month') AS ds,
               ROUND(SUM(total_amount)/1e6, 2) AS revenue_m
        FROM orders WHERE status='Delivered'
        GROUP BY ds ORDER BY ds
    """, conn)
    conn.close()
    monthly["ds"] = pd.to_datetime(monthly["ds"])
    monthly["t"]  = np.arange(len(monthly))
    monthly["sin12"] = np.sin(2*np.pi*monthly["t"]/12)
    monthly["cos12"] = np.cos(2*np.pi*monthly["t"]/12)
    monthly["is_festive"] = monthly["ds"].dt.month.isin([10,11,12]).astype(int)
    monthly["is_summer"]  = monthly["ds"].dt.month.isin([4,5]).astype(int)

    from sklearn.linear_model import Ridge
    feat = ["t","sin12","cos12","is_festive","is_summer"]
    model = Ridge(alpha=1.0).fit(monthly[feat], monthly["revenue_m"])
    monthly["fitted"] = model.predict(monthly[feat])
    residual_std = (monthly["revenue_m"] - monthly["fitted"]).std()

    # future 6 months
    future = []
    for i in range(1, 7):
        t_val = monthly["t"].max() + i
        ds    = monthly["ds"].max() + pd.DateOffset(months=i)
        m     = ds.month
        row   = {"t": t_val, "ds": ds,
                 "sin12": np.sin(2*np.pi*t_val/12),
                 "cos12": np.cos(2*np.pi*t_val/12),
                 "is_festive": int(m in [10,11,12]),
                 "is_summer":  int(m in [4,5])}
        future.append(row)
    future_df = pd.DataFrame(future)
    future_df["forecast"] = model.predict(future_df[feat])
    future_df["lower"]    = future_df["forecast"] - 1.64 * residual_std
    future_df["upper"]    = future_df["forecast"] + 1.64 * residual_std
    return monthly, future_df


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📦 RetailPulse")
    st.markdown("*Executive Analytics Dashboard*")
    st.divider()

    customers, products, orders, order_items, sessions = load_all()

    st.markdown("**Filters**")
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    min_date = orders["order_date"].min().date()
    max_date = orders["order_date"].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        key="date_filter"
    )

    # Handle single-date and range selection
    if isinstance(date_range, tuple):
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range[0]
    else:
        start_date = end_date = date_range

    selected_cities = st.multiselect(
        "Cities",
        options=sorted(customers["city"].unique().tolist()),
        default=[],
        placeholder="All cities",
    )

    st.divider()
    st.markdown("**Dataset**")
    st.caption(f"🧑 {len(customers):,} customers")
    st.caption(f"📦 {len(orders):,} orders")
    st.caption(f"🛒 {len(sessions):,} sessions")
    st.caption(f"🗓 Jan 2023 – Jun 2024")


# ── Apply filters ─────────────────────────────────────────────
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

filt_orders = orders[
    (orders["order_date"].dt.date >= start_date) &
    (orders["order_date"].dt.date <= end_date)
]
if selected_cities:
    city_custs = customers[customers["city"].isin(selected_cities)]["customer_id"]
    filt_orders = filt_orders[filt_orders["customer_id"].isin(city_custs)]


# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════

st.title("📦 RetailPulse — Analytics Dashboard")
st.caption("E-commerce business intelligence · Jan 2023 – Jun 2024 · INR")
st.divider()


# ═══════════════════════════════════════════════════════════════
#  KPI CARDS — TOP ROW
# ═══════════════════════════════════════════════════════════════

del_orders = filt_orders[filt_orders["status"] == "Delivered"]
total_rev  = del_orders["total_amount"].sum()
n_orders   = len(del_orders)
aov        = del_orders["total_amount"].mean() if n_orders > 0 else 0
cancel_rate= (filt_orders["status"]=="Cancelled").mean() * 100
unique_custs= filt_orders["customer_id"].nunique()

# compare to prev period for delta
if isinstance(date_range, tuple) and len(date_range) == 2:
    period_days = max((date_range[1] - date_range[0]).days, 1)
    current_start = date_range[0]
else:
    period_days = 1
    current_start = start_date
prev_start = current_start - pd.Timedelta(days=period_days)
prev_orders = orders[
    (orders["order_date"].dt.date >= prev_start) &
    (orders["order_date"].dt.date < date_range[0]) &
    (orders["status"] == "Delivered")
]
prev_rev = prev_orders["total_amount"].sum() or 1
rev_delta = (total_rev - prev_rev) / prev_rev * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Delivered Revenue",    f"Rs{total_rev/1e6:.1f}M",  f"{rev_delta:+.1f}% vs prev period")
col2.metric("🛍 Delivered Orders",     f"{n_orders:,}",            f"{n_orders - len(prev_orders):+,}")
col3.metric("🎯 Avg Order Value",      f"Rs{aov:,.0f}",            "")
col4.metric("❌ Cancellation Rate",    f"{cancel_rate:.1f}%",      "")
col5.metric("👥 Unique Customers",     f"{unique_custs:,}",        "")

st.markdown("")


# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Revenue",
    "🛒 Products",
    "👥 Customers",
    "🌐 Funnel",
    "🔮 Forecast",
])


# ───────────────────────────────────────────────────────────────
# TAB 1 — REVENUE
# ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Revenue Trends")

    monthly = monthly_revenue()

    # Monthly revenue line
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["revenue_m"],
        name="Gross Revenue (Rs M)", marker_color="#BFDBFE",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["del_rev_m"],
        name="Delivered Revenue (Rs M)", line=dict(color="#2563EB", width=3),
        mode="lines+markers", marker_size=6,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["mom_growth"],
        name="MoM Growth %", line=dict(color="#DC2626", width=2, dash="dot"),
        mode="lines+markers", marker_size=5,
    ), secondary_y=True)

    fig.update_layout(
        title="Monthly Revenue & Growth Rate",
        height=380, template="plotly_white",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        yaxis_title="Revenue (Rs M)",
        yaxis2_title="MoM Growth %",
    )
    fig.update_yaxes(title_text="Revenue (Rs M)", secondary_y=False)
    fig.update_yaxes(title_text="MoM Growth %", secondary_y=True, ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    # Category revenue + Payment method
    col_a, col_b = st.columns(2)

    with col_a:
        cat = category_revenue()
        fig2 = px.bar(cat, x="revenue_m", y="category", orientation="h",
                      title="Revenue by Category (Rs M)", color="revenue_m",
                      color_continuous_scale="Blues",
                      labels={"revenue_m":"Revenue (Rs M)","category":""},
                      height=340, text="revenue_m")
        fig2.update_traces(texttemplate="Rs%{text}M", textposition="outside")
        fig2.update_layout(template="plotly_white", coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        pay = (filt_orders[filt_orders["status"]=="Delivered"]
               .groupby("payment_method")["total_amount"].sum()
               .reset_index())
        pay["rev_m"] = pay["total_amount"] / 1e6
        fig3 = px.pie(pay, values="rev_m", names="payment_method",
                      title="Revenue by Payment Method",
                      color_discrete_sequence=PALETTE, hole=0.45, height=340)
        fig3.update_traces(textinfo="percent+label")
        fig3.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # City revenue map (bar chart)
    city_rev = (filt_orders[filt_orders["status"]=="Delivered"]
                .merge(customers[["customer_id","city"]], on="customer_id")
                .groupby("city")["total_amount"].sum()
                .nlargest(10).reset_index())
    city_rev["rev_m"] = city_rev["total_amount"] / 1e6
    fig4 = px.bar(city_rev, x="city", y="rev_m",
                  title="Top 10 Cities by Delivered Revenue",
                  color="rev_m", color_continuous_scale="Blues",
                  labels={"rev_m":"Revenue (Rs M)","city":""},
                  height=320, text_auto=".1f")
    fig4.update_traces(texttemplate="Rs%{y:.1f}M", textposition="outside")
    fig4.update_layout(template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — PRODUCTS
# ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Product Performance")

    top_products = pd.read_sql("""
        SELECT p.name, p.category, p.price, p.rating,
               SUM(oi.quantity) AS units_sold,
               ROUND(SUM(oi.line_total)/1000,1) AS revenue_k,
               ROUND(AVG(oi.discount_pct)*100,1) AS avg_discount_pct
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN orders   o ON oi.order_id   = o.order_id
        WHERE o.status = 'Delivered'
        GROUP BY p.product_id
        ORDER BY revenue_k DESC LIMIT 20
    """, sqlite3.connect(DB_PATH))

    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig = px.scatter(top_products, x="units_sold", y="revenue_k",
                         size="price", color="category",
                         hover_name="name",
                         title="Top 20 Products — Units vs Revenue (bubble = price)",
                         labels={"units_sold":"Units Sold","revenue_k":"Revenue (Rs K)"},
                         color_discrete_sequence=PALETTE, height=400)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Price vs Rating scatter
        active_prods = products[products["is_active"]==1].copy()
        fig2 = px.scatter(active_prods, x="price", y="rating",
                          color="category", size="review_count",
                          title="Price vs Rating (size = reviews)",
                          labels={"price":"Price (Rs)","rating":"Rating"},
                          color_discrete_sequence=PALETTE,
                          log_x=True, height=400)
        fig2.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Category discount analysis
    disc = pd.read_sql("""
        SELECT p.category,
               ROUND(AVG(oi.discount_pct)*100,1) AS avg_discount,
               ROUND(SUM(oi.line_total)/1e6,2)   AS revenue_m
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN orders   o ON oi.order_id   = o.order_id
        WHERE o.status='Delivered'
        GROUP BY p.category ORDER BY avg_discount DESC
    """, sqlite3.connect(DB_PATH))

    fig3 = px.bar(disc, x="category", y="avg_discount",
                  color="revenue_m", color_continuous_scale="RdYlGn_r",
                  title="Avg Discount % by Category (color = revenue)",
                  labels={"avg_discount":"Avg Discount (%)","category":""},
                  text_auto=".1f", height=320)
    fig3.update_traces(texttemplate="%{y}%", textposition="outside")
    fig3.update_layout(template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 20 Products Table")
    st.dataframe(
        top_products.rename(columns={
            "name":"Product","category":"Category","price":"Price (Rs)",
            "rating":"Rating","units_sold":"Units Sold",
            "revenue_k":"Revenue (Rs K)","avg_discount_pct":"Avg Discount %"
        }),
        use_container_width=True, hide_index=True,
    )


# ───────────────────────────────────────────────────────────────
# TAB 3 — CUSTOMERS
# ───────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Customer Intelligence")

    rfm = build_rfm()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        seg_count = rfm["rfm_segment"].value_counts().reset_index()
        seg_count.columns = ["Segment","Customers"]
        fig = px.pie(seg_count, values="Customers", names="Segment",
                     title="RFM Segments", hole=0.5,
                     color_discrete_sequence=PALETTE, height=320)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        acq = rfm["acquisition_channel"].value_counts().reset_index()
        acq.columns = ["Channel","Customers"]
        fig2 = px.bar(acq, x="Customers", y="Channel", orientation="h",
                      title="Customers by Channel",
                      color="Customers", color_continuous_scale="Blues",
                      height=320)
        fig2.update_layout(template="plotly_white", coloraxis_showscale=False,
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True)

    with col_c:
        fig3 = px.histogram(rfm, x="age", nbins=25,
                            title="Customer Age Distribution",
                            color_discrete_sequence=["#2563EB"], height=320)
        fig3.update_layout(template="plotly_white",
                           xaxis_title="Age", yaxis_title="Customers")
        st.plotly_chart(fig3, use_container_width=True)

    # RFM heatmap: Recency vs Frequency
    st.subheader("RFM Heatmap — Recency × Frequency")
    heat = rfm.groupby(["r","f"])["monetary"].mean().reset_index()
    heat_pivot = heat.pivot(index="r", columns="f", values="monetary")
    fig4 = px.imshow(heat_pivot, color_continuous_scale="Blues",
                     title="Avg Monetary by R-score (rows) and F-score (cols)",
                     labels=dict(x="Frequency Score", y="Recency Score", color="Avg Revenue (Rs)"),
                     text_auto=".0f", height=340)
    fig4.update_layout(template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    # Churn risk from RFM
    col_d, col_e = st.columns(2)
    with col_d:
        city_seg = rfm.groupby(["city","rfm_segment"]).size().reset_index(name="customers")
        fig5 = px.bar(city_seg, x="city", y="customers", color="rfm_segment",
                      title="Customer Segments by City", barmode="stack",
                      color_discrete_sequence=PALETTE, height=360)
        fig5.update_layout(template="plotly_white",
                           xaxis_tickangle=-30, legend_title="Segment")
        st.plotly_chart(fig5, use_container_width=True)

    with col_e:
        fig6 = px.box(rfm, x="rfm_segment", y="monetary",
                      title="Revenue Distribution by RFM Segment",
                      color="rfm_segment", color_discrete_sequence=PALETTE,
                      height=360, log_y=True)
        fig6.update_layout(template="plotly_white", showlegend=False,
                           xaxis_title="", yaxis_title="Total Revenue (Rs, log)")
        st.plotly_chart(fig6, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 4 — FUNNEL
# ───────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Web Conversion Funnel")

    funnel_df, sess_df = funnel_data()

    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig = go.Figure(go.Funnel(
            y=funnel_df["stage"],
            x=funnel_df["reached"],
            textinfo="value+percent initial",
            marker=dict(color=PALETTE[:len(funnel_df)]),
        ))
        fig.update_layout(title="Session Conversion Funnel",
                          height=420, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Device conversion
        dev = (sess_df.groupby("device")
               .agg(sessions=("converted","count"),
                    conversions=("converted","sum"))
               .reset_index())
        dev["cvr"] = dev["conversions"] / dev["sessions"] * 100

        fig2 = px.bar(dev, x="device", y="cvr",
                      title="Conversion Rate by Device",
                      color="device", text_auto=".2f",
                      color_discrete_sequence=PALETTE, height=220)
        fig2.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
        fig2.update_layout(template="plotly_white", showlegend=False,
                           yaxis_title="CVR (%)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

        # Channel conversion
        ch = (sess_df.groupby("channel")
              .agg(sessions=("converted","count"),
                   conversions=("converted","sum"))
              .reset_index())
        ch["cvr"] = ch["conversions"] / ch["sessions"] * 100
        ch = ch.sort_values("cvr", ascending=False)

        fig3 = px.bar(ch, x="channel", y="cvr",
                      title="Conversion Rate by Channel",
                      color="cvr", color_continuous_scale="Blues",
                      text_auto=".2f", height=210)
        fig3.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
        fig3.update_layout(template="plotly_white", coloraxis_showscale=False,
                           yaxis_title="CVR (%)", xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    # Stage drop-off waterfall
    st.subheader("Stage Drop-off Analysis")
    funnel_df["drop_pct"] = funnel_df["drop_pct"].fillna(0)
    funnel_df["keep"] = funnel_df["reached"] / funnel_df["reached"].iloc[0] * 100

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=funnel_df["stage"], y=funnel_df["keep"],
        name="% Reaching Stage", marker_color="#2563EB", text=funnel_df["keep"].round(1),
        texttemplate="%{text:.1f}%", textposition="outside",
    ))
    fig4.add_trace(go.Scatter(
        x=funnel_df["stage"], y=funnel_df["drop_pct"].abs(),
        name="Drop-off %", line=dict(color="#DC2626", width=2.5),
        mode="lines+markers", marker_size=8, yaxis="y2",
    ))
    fig4.update_layout(
        title="Funnel Stage Retention & Drop-off",
        height=360, template="plotly_white",
        yaxis=dict(title="% Reaching Stage", range=[0, 120]),
        yaxis2=dict(title="Drop-off %", overlaying="y", side="right",
                    ticksuffix="%", showgrid=False),
        legend=dict(orientation="h", y=1.08), hovermode="x unified"
    )
    st.plotly_chart(fig4, use_container_width=True)


# ───────────────────────────────────────────────────────────────
# TAB 5 — FORECAST
# ───────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Sales Forecast — Next 6 Months")

    monthly_df, future_df = forecast_data()

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=monthly_df["ds"], y=monthly_df["revenue_m"],
        name="Historical", line=dict(color="#2563EB", width=2.5),
        mode="lines+markers", marker_size=5,
    ))

    # Model fit
    fig.add_trace(go.Scatter(
        x=monthly_df["ds"], y=monthly_df["fitted"],
        name="Model Fit", line=dict(color="#059669", width=2, dash="dash"),
        mode="lines",
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([future_df["ds"], future_df["ds"][::-1]]),
        y=pd.concat([future_df["upper"], future_df["lower"][::-1]]),
        fill="toself", fillcolor="rgba(217,119,6,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence Interval", showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_df["ds"], y=future_df["forecast"],
        name="Forecast", line=dict(color="#D97706", width=3),
        mode="lines+markers", marker_size=9, marker_symbol="diamond",
    ))

    fig.add_vline(x=monthly_df["ds"].max(), line_dash="dot",
                  line_color="#888", annotation_text="Forecast →",
                  annotation_position="top right")

    fig.update_layout(
        title="Revenue Forecast with 90% Confidence Interval",
        height=440, template="plotly_white",
        yaxis_title="Revenue (Rs M)",
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.subheader("Monthly Forecast Table")
    display = future_df[["ds","forecast","lower","upper"]].copy()
    display.columns = ["Month","Forecast (Rs M)","Lower 90% (Rs M)","Upper 90% (Rs M)"]
    display["Month"] = display["Month"].dt.strftime("%B %Y")
    for col in ["Forecast (Rs M)","Lower 90% (Rs M)","Upper 90% (Rs M)"]:
        display[col] = display[col].map(lambda x: f"Rs {x:.1f}M")

    st.dataframe(display, use_container_width=True, hide_index=True)

    col_a, col_b, col_c = st.columns(3)
    total_fcst = future_df["forecast"].sum()
    festive_fcst = future_df[future_df["ds"].dt.month.isin([10,11,12])]["forecast"].sum()
    col_a.metric("📊 Total 6M Forecast",   f"Rs{total_fcst:.1f}M")
    col_b.metric("🎉 Festive Quarter (Q4)", f"Rs{festive_fcst:.1f}M")
    col_c.metric("📈 Dec 2024 Peak",        f"Rs{future_df['forecast'].max():.1f}M")


# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "RetailPulse Analytics Dashboard · Built with Streamlit + Plotly · "
    "Data: synthetic e-commerce (2023–2024) · "
    "github.com/yourname/retailpulse"
)
