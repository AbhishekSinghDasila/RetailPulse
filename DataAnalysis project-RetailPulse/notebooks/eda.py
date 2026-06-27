"""
RetailPulse — Phase 3: Exploratory Data Analysis
=================================================
Run this script to generate 12 publication-quality charts
saved to notebooks/charts/

Usage:
    python notebooks/eda.py

Requirements:
    pip install pandas numpy matplotlib seaborn
"""

import os
import sqlite3
import warnings

import matplotlib
matplotlib.use("Agg")           # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── setup ─────────────────────────────────────────────────────
DB_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "retailpulse.db")
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

PALETTE    = ["#2563EB","#7C3AED","#059669","#D97706","#DC2626","#0891B2","#BE185D","#65A30D"]
BLUE       = "#2563EB"
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False})

def save(name):
    path = os.path.join(CHARTS_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


# ── load data ─────────────────────────────────────────────────
def load():
    conn = sqlite3.connect(DB_PATH)
    customers   = pd.read_sql("SELECT * FROM customers",   conn, parse_dates=["signup_date"])
    products    = pd.read_sql("SELECT * FROM products",    conn)
    orders      = pd.read_sql("SELECT * FROM orders",      conn, parse_dates=["order_date","delivered_at"])
    order_items = pd.read_sql("SELECT * FROM order_items", conn)
    sessions    = pd.read_sql("SELECT * FROM sessions",    conn, parse_dates=["session_start"])
    conn.close()
    return customers, products, orders, order_items, sessions


# ═══════════════════════════════════════════════════════════════
# CHART 1  Monthly Revenue Trend
# ═══════════════════════════════════════════════════════════════
def chart_monthly_revenue(orders):
    print("\n[1/12] Monthly revenue trend")
    df = (
        orders[orders["status"] == "Delivered"]
        .set_index("order_date")
        .resample("ME")["total_amount"]
        .sum()
        .reset_index()
    )
    df["month_label"] = df["order_date"].dt.strftime("%b %Y")
    df["rev_m"]       = df["total_amount"] / 1e6

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(df.index, df["rev_m"], alpha=0.15, color=BLUE)
    ax.plot(df.index, df["rev_m"], color=BLUE, lw=2.5, marker="o", ms=5)

    # annotate festive peak
    peak_idx = df["rev_m"].idxmax()
    ax.annotate(
        f"Festive peak\nRs{df.loc[peak_idx,'rev_m']:.1f}M",
        xy=(peak_idx, df.loc[peak_idx, "rev_m"]),
        xytext=(peak_idx - 2, df.loc[peak_idx, "rev_m"] * 0.82),
        arrowprops=dict(arrowstyle="->", color="#555"),
        fontsize=9, color="#333"
    )

    ax.set_xticks(df.index[::2])
    ax.set_xticklabels(df["month_label"].iloc[::2], rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:.0f}M"))
    ax.set_title("Monthly Delivered Revenue (Jan 2023 – Jun 2024)", fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Revenue")
    save("01_monthly_revenue.png")


# ═══════════════════════════════════════════════════════════════
# CHART 2  Revenue by Category (horizontal bar)
# ═══════════════════════════════════════════════════════════════
def chart_category_revenue(orders, order_items, products):
    print("[2/12] Revenue by category")
    df = (
        order_items
        .merge(orders[orders["status"]=="Delivered"][["order_id"]], on="order_id")
        .merge(products[["product_id","category"]], on="product_id")
        .groupby("category")["line_total"]
        .sum()
        .sort_values()
        .reset_index()
    )
    df["rev_m"] = df["line_total"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["category"], df["rev_m"], color=PALETTE[:len(df)], height=0.6)
    for bar, val in zip(bars, df["rev_m"]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f"Rs{val:.1f}M", va="center", fontsize=9)
    ax.set_xlabel("Revenue (Rs M)")
    ax.set_title("Revenue by Category — Delivered Orders", fontsize=14, pad=12)
    save("02_category_revenue.png")


# ═══════════════════════════════════════════════════════════════
# CHART 3  Order Status Distribution (donut)
# ═══════════════════════════════════════════════════════════════
def chart_order_status(orders):
    print("[3/12] Order status donut")
    counts = orders["status"].value_counts()
    colors = ["#059669","#2563EB","#D97706","#DC2626","#7C3AED"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Order Status Distribution\n(52,000 orders)", fontsize=14, pad=18)
    save("03_order_status.png")


# ═══════════════════════════════════════════════════════════════
# CHART 4  Customer Age Distribution by Gender
# ═══════════════════════════════════════════════════════════════
def chart_age_gender(customers):
    print("[4/12] Age distribution by gender")
    df = customers[customers["gender"].isin(["M","F"])]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for gender, color, label in [("M","#2563EB","Male"), ("F","#BE185D","Female")]:
        subset = df[df["gender"]==gender]["age"]
        ax.hist(subset, bins=20, alpha=0.65, color=color, label=label, edgecolor="white")

    ax.axvline(df["age"].mean(), color="#555", ls="--", lw=1.5, label=f"Mean {df['age'].mean():.1f}y")
    ax.set_xlabel("Age")
    ax.set_ylabel("Customers")
    ax.set_title("Customer Age Distribution by Gender", fontsize=14, pad=12)
    ax.legend()
    save("04_age_distribution.png")


# ═══════════════════════════════════════════════════════════════
# CHART 5  Customer Segment Breakdown
# ═══════════════════════════════════════════════════════════════
def chart_segments(customers):
    print("[5/12] Customer segments")
    seg = customers["segment"].value_counts().reset_index()
    seg.columns = ["segment","count"]
    seg["pct"] = (seg["count"] / seg["count"].sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(seg["segment"], seg["count"],
                  color=PALETTE[:len(seg)], width=0.55, edgecolor="white")
    for bar, row in zip(bars, seg.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{row.pct}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Customers")
    ax.set_title("Customer Segment Distribution", fontsize=14, pad=12)
    save("05_customer_segments.png")


# ═══════════════════════════════════════════════════════════════
# CHART 6  Acquisition Channel vs Revenue
# ═══════════════════════════════════════════════════════════════
def chart_channel_revenue(orders, customers):
    print("[6/12] Channel revenue")
    df = (
        orders[orders["status"]=="Delivered"]
        .merge(customers[["customer_id","acquisition_channel"]], on="customer_id")
        .groupby("acquisition_channel")
        .agg(revenue=("total_amount","sum"), orders=("order_id","count"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    df["rev_m"]     = df["revenue"] / 1e6
    df["aov"]       = (df["revenue"] / df["orders"]).round(0)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    bars = ax1.bar(df["acquisition_channel"], df["rev_m"],
                   color=PALETTE[:len(df)], width=0.5, edgecolor="white", label="Revenue")
    ax2.plot(df["acquisition_channel"], df["aov"], color="#DC2626",
             marker="D", ms=8, lw=2, label="AOV (Rs)")

    ax1.set_ylabel("Revenue (Rs M)")
    ax2.set_ylabel("Avg Order Value (Rs)", color="#DC2626")
    ax2.tick_params(axis="y", colors="#DC2626")
    ax1.set_title("Acquisition Channel — Revenue & Average Order Value", fontsize=14, pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
    save("06_channel_revenue.png")


# ═══════════════════════════════════════════════════════════════
# CHART 7  Web Funnel Drop-off
# ═══════════════════════════════════════════════════════════════
def chart_funnel(sessions):
    print("[7/12] Web funnel")
    PAGE_ORDER = ["home","category","product","cart","checkout","confirmation"]
    # sessions that REACHED each stage = sessions with pages_viewed >= stage depth
    reached = []
    for depth, page in enumerate(PAGE_ORDER, 1):
        reached.append({
            "stage": page.title(),
            "sessions": (sessions["pages_viewed"] >= depth).sum()
        })
    df = pd.DataFrame(reached)
    df["pct"] = (df["sessions"] / df.loc[0,"sessions"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [f"#{int(50+i*35):02X}98EB" for i in range(len(df))]
    bars = ax.barh(df["stage"][::-1], df["sessions"][::-1],
                   color=PALETTE[::-1][:len(df)], height=0.55)
    for bar, row in zip(bars, df[::-1].itertuples()):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f"{row.pct:.1f}%  ({row.sessions:,})",
                va="center", fontsize=9.5)
    ax.set_xlabel("Sessions Reaching Stage")
    ax.set_title("Website Conversion Funnel", fontsize=14, pad=12)
    ax.set_xlim(0, df["sessions"].max() * 1.25)
    save("07_funnel.png")


# ═══════════════════════════════════════════════════════════════
# CHART 8  Device Split — Sessions & Conversion Rate
# ═══════════════════════════════════════════════════════════════
def chart_device(sessions):
    print("[8/12] Device split")
    df = (
        sessions.groupby("device")
        .agg(sessions=("session_id","count"), converted=("converted","sum"))
        .reset_index()
    )
    df["cvr"] = df["converted"] / df["sessions"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.pie(df["sessions"], labels=df["device"], autopct="%1.1f%%",
            colors=PALETTE[:3], startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
    ax1.set_title("Session Share by Device", fontsize=12)

    bars = ax2.bar(df["device"], df["cvr"], color=PALETTE[:3], width=0.45, edgecolor="white")
    for bar, val in zip(bars, df["cvr"]):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.05,
                 f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Conversion Rate (%)")
    ax2.set_title("Conversion Rate by Device", fontsize=12)
    ax2.set_ylim(0, df["cvr"].max() * 1.25)

    plt.suptitle("Device Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    save("08_device_analysis.png")


# ═══════════════════════════════════════════════════════════════
# CHART 9  Payment Method Mix
# ═══════════════════════════════════════════════════════════════
def chart_payment(orders):
    print("[9/12] Payment methods")
    df = (
        orders[orders["status"]=="Delivered"]
        .groupby("payment_method")
        .agg(orders=("order_id","count"), revenue=("total_amount","sum"))
        .reset_index()
        .sort_values("orders", ascending=False)
    )
    df["rev_m"] = df["revenue"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(df["payment_method"], df["orders"],
                  color=PALETTE[:len(df)], width=0.5, edgecolor="white")
    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+40,
                f"Rs{row.rev_m:.0f}M", ha="center", fontsize=9, color="#555")
    ax.set_ylabel("Delivered Orders")
    ax.set_title("Payment Method — Order Count & Revenue (label)", fontsize=14, pad=12)
    save("09_payment_methods.png")


# ═══════════════════════════════════════════════════════════════
# CHART 10 Top 10 Cities by Revenue
# ═══════════════════════════════════════════════════════════════
def chart_cities(orders, customers):
    print("[10/12] Cities revenue")
    df = (
        orders[orders["status"]=="Delivered"]
        .merge(customers[["customer_id","city"]], on="customer_id")
        .groupby("city")["total_amount"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    df["rev_m"] = df["total_amount"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["city"][::-1], df["rev_m"][::-1],
                   color=PALETTE[:len(df)], height=0.55)
    for bar, val in zip(bars, df["rev_m"][::-1]):
        ax.text(val + 0.3, bar.get_y()+bar.get_height()/2,
                f"Rs{val:.1f}M", va="center", fontsize=9)
    ax.set_xlabel("Revenue (Rs M)")
    ax.set_title("Top 10 Cities by Delivered Revenue", fontsize=14, pad=12)
    save("10_city_revenue.png")


# ═══════════════════════════════════════════════════════════════
# CHART 11 Product Price Distribution by Category (boxplot)
# ═══════════════════════════════════════════════════════════════
def chart_price_boxplot(products):
    print("[11/12] Price distribution boxplot")
    df = products[products["is_active"]==1].copy()
    # log scale for wide range
    df["log_price"] = np.log10(df["price"])
    order = df.groupby("category")["price"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="category", y="log_price", order=order,
                palette=PALETTE[:len(order)], width=0.5, linewidth=1.2, ax=ax)

    yticks = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"Rs{10**y:,.0f}" for y in yticks])
    ax.set_xlabel("")
    ax.set_ylabel("Price (log scale)")
    ax.set_title("Product Price Distribution by Category", fontsize=14, pad=12)
    plt.xticks(rotation=20, ha="right")
    save("11_price_boxplot.png")


# ═══════════════════════════════════════════════════════════════
# CHART 12 Discount vs Revenue Scatter (sample)
# ═══════════════════════════════════════════════════════════════
def chart_discount_scatter(order_items, orders, products):
    print("[12/12] Discount vs line total scatter")
    df = (
        order_items
        .sample(5_000, random_state=42)
        .merge(products[["product_id","category"]], on="product_id")
        .merge(orders[["order_id","status"]], on="order_id")
        .query("status == 'Delivered'")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (cat, grp) in enumerate(df.groupby("category")):
        ax.scatter(grp["discount_pct"]*100, grp["line_total"],
                   alpha=0.35, s=18, color=PALETTE[i % len(PALETTE)], label=cat)

    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Line Total (Rs)")
    ax.set_title("Discount vs Line Revenue — Delivered Items (sample 5k)", fontsize=14, pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"Rs{x:,.0f}"))
    ax.legend(title="Category", fontsize=8, title_fontsize=9,
              loc="upper right", ncol=2, framealpha=0.85)
    save("12_discount_scatter.png")


# ═══════════════════════════════════════════════════════════════
# Data Quality Report
# ═══════════════════════════════════════════════════════════════
def data_quality_report(customers, products, orders, order_items, sessions):
    print("\n" + "="*55)
    print("  DATA QUALITY REPORT")
    print("="*55)

    frames = {
        "customers":   customers,
        "products":    products,
        "orders":      orders,
        "order_items": order_items,
        "sessions":    sessions,
    }
    for name, df in frames.items():
        nulls  = df.isnull().sum().sum()
        dupes  = df.duplicated().sum()
        print(f"\n  [{name}]  shape={df.shape}  nulls={nulls}  duplicates={dupes}")

    # Business rules
    bad_dates = (orders["order_date"].dt.date < pd.to_datetime(customers.set_index("customer_id")
                 .loc[orders["customer_id"].values, "signup_date"].values).date).sum()
    print(f"\n  Orders before customer signup : {bad_dates} (expect 0)")

    neg_price  = (products["price"] <= 0).sum()
    neg_cost   = (products["cost"]  <= 0).sum()
    print(f"  Products with invalid price   : {neg_price}")
    print(f"  Products with invalid cost    : {neg_cost}")

    margin = ((products["price"] - products["cost"]) / products["price"] * 100)
    print(f"  Avg gross margin              : {margin.mean():.1f}%  (min {margin.min():.1f}%)")
    print("="*55)


# ── main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("RetailPulse — Phase 3: EDA\n")
    print("Loading data...")
    customers, products, orders, order_items, sessions = load()
    print(f"  customers={len(customers):,}  products={len(products):,}  "
          f"orders={len(orders):,}  items={len(order_items):,}  sessions={len(sessions):,}\n")

    print("Generating charts:")
    chart_monthly_revenue(orders)
    chart_category_revenue(orders, order_items, products)
    chart_order_status(orders)
    chart_age_gender(customers)
    chart_segments(customers)
    chart_channel_revenue(orders, customers)
    chart_funnel(sessions)
    chart_device(sessions)
    chart_payment(orders)
    chart_cities(orders, customers)
    chart_price_boxplot(products)
    chart_discount_scatter(order_items, orders, products)

    data_quality_report(customers, products, orders, order_items, sessions)

    print(f"\n  All 12 charts saved to notebooks/charts/")
    print("  Next step -> Phase 4: Machine Learning (models/)")
