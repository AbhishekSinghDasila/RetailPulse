"""
RetailPulse — Phase 4: Machine Learning Models
===============================================
Three models in one script:

  Model 1 — Churn Prediction      (Random Forest classifier)
  Model 2 — Customer Segmentation (K-Means clustering)
  Model 3 — Sales Forecasting     (Linear trend + seasonality)

Usage:
    pip install scikit-learn pandas numpy matplotlib
    python models/ml_models.py

Outputs:
    models/charts/  — evaluation plots
    models/reports/ — metrics as text
"""

import os
import sqlite3
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, roc_curve,
                                     silhouette_score)
from sklearn.cluster         import KMeans
from sklearn.decomposition   import PCA
from sklearn.pipeline        import Pipeline

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
DB_PATH    = os.path.join(BASE, "..", "data", "retailpulse.db")
CHARTS_DIR = os.path.join(BASE, "charts")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────
PALETTE = ["#2563EB","#7C3AED","#059669","#D97706","#DC2626","#0891B2","#BE185D","#65A30D"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

def save(name):
    path = os.path.join(CHARTS_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"    Chart → {path}")

def divider(title):
    w = 60
    print("\n" + "═"*w)
    print(f"  {title}")
    print("═"*w)


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — shared across all models
# ══════════════════════════════════════════════════════════════

def build_customer_features():
    """
    Build one row per customer with RFM + behavioural features.
    This is the single most important function — interviewers
    always ask 'what features did you engineer?'
    """
    conn = sqlite3.connect(DB_PATH)
    SNAPSHOT = "2024-06-30"   # pretend today is the end of the dataset

    # ── RFM from delivered orders ──
    rfm = pd.read_sql(f"""
        SELECT
            o.customer_id,
            CAST(julianday('{SNAPSHOT}') - julianday(MAX(o.order_date)) AS INTEGER)
                                                        AS recency_days,
            COUNT(DISTINCT o.order_id)                  AS frequency,
            ROUND(SUM(o.total_amount), 2)               AS monetary,
            ROUND(AVG(o.total_amount), 2)               AS avg_order_value,
            ROUND(MAX(o.total_amount), 2)               AS max_order_value,
            ROUND(MIN(o.total_amount), 2)               AS min_order_value,
            SUM(CASE WHEN o.status='Cancelled'  THEN 1 ELSE 0 END) AS cancellations,
            SUM(CASE WHEN o.status='Returned'   THEN 1 ELSE 0 END) AS returns,
            SUM(o.shipping_fee)                         AS total_shipping_paid,
            COUNT(DISTINCT strftime('%Y-%m', o.order_date)) AS active_months
        FROM orders o
        GROUP BY o.customer_id
    """, conn)

    # ── Category diversity ──
    cat_div = pd.read_sql(f"""
        SELECT oi.order_id, o.customer_id,
               COUNT(DISTINCT p.category) AS n_categories
        FROM order_items oi
        JOIN orders   o ON oi.order_id   = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        GROUP BY o.customer_id
    """, conn)
    cat_div = cat_div.groupby("customer_id")["n_categories"].mean().reset_index()
    cat_div.columns = ["customer_id","avg_categories_per_order"]

    # ── Session behaviour ──
    sessions = pd.read_sql("""
        SELECT customer_id,
               COUNT(*)                             AS session_count,
               ROUND(AVG(duration_seconds)/60.0,1) AS avg_session_min,
               ROUND(AVG(pages_viewed),1)          AS avg_pages,
               SUM(converted)                      AS total_conversions,
               COUNT(DISTINCT device)              AS devices_used
        FROM sessions
        WHERE customer_id IS NOT NULL
        GROUP BY customer_id
    """, conn)

    # ── Customer demographics ──
    customers = pd.read_sql("""
        SELECT customer_id, age, gender, city,
               segment, acquisition_channel,
               CAST(julianday('2024-06-30') - julianday(signup_date) AS INTEGER)
                   AS tenure_days,
               is_active
        FROM customers
    """, conn)
    conn.close()

    # ── Merge everything ──
    df = (customers
          .merge(rfm,      on="customer_id", how="left")
          .merge(cat_div,  on="customer_id", how="left")
          .merge(sessions, on="customer_id", how="left"))

    df.fillna(0, inplace=True)

    # ── Derived features ──
    df["cancel_rate"]      = df["cancellations"] / df["frequency"].clip(1)
    df["return_rate"]      = df["returns"]       / df["frequency"].clip(1)
    df["revenue_per_month"]= df["monetary"]      / df["active_months"].clip(1)
    df["conversion_rate"]  = df["total_conversions"] / df["session_count"].clip(1)
    df["order_spread"]     = df["max_order_value"] - df["min_order_value"]

    # ── Churn label: customers with no order in last 30 days (~40% rate) ──
    df["churned"] = (df["recency_days"] > 30).astype(int)

    return df


# ══════════════════════════════════════════════════════════════
#  MODEL 1 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════

def model_churn(df):
    divider("MODEL 1 — Customer Churn Prediction (Random Forest)")

    FEATURES = [
        "recency_days","frequency","monetary","avg_order_value",
        "cancellations","returns","cancel_rate","return_rate",
        "active_months","revenue_per_month","avg_categories_per_order",
        "session_count","avg_session_min","avg_pages","devices_used",
        "conversion_rate","tenure_days","age","total_shipping_paid","order_spread",
    ]

    X = df[FEATURES].values
    y = df["churned"].values

    print(f"  Churn rate: {y.mean()*100:.1f}%  ({y.sum():,} / {len(y):,} customers)")
    print(f"  Features  : {len(FEATURES)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train ──
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    cv = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="roc_auc")

    print(f"\n  Test AUC        : {auc:.4f}")
    print(f"  5-Fold CV AUC   : {cv.mean():.4f} ± {cv.std():.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained","Churned"]))

    # ── Feature importance chart ──
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    top12 = importances.nlargest(12).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top12.index, top12.values, color="#2563EB", height=0.6)
    for bar, val in zip(bars, top12.values):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Churn Model — Top 12 Feature Importances\n(Random Forest, 300 trees)", pad=12)
    save("M1_feature_importance.png")

    # ── ROC curve ──
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"RF (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction", pad=12)
    ax.legend(fontsize=11)
    save("M1_roc_curve.png")

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Retained","Churned"])
    ax.set_yticklabels(["Retained","Churned"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Churn Model", pad=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i,j] > cm.max()//2 else "black")
    save("M1_confusion_matrix.png")

    # ── Churn risk score distribution ──
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(y_proba[y_test==0], bins=40, alpha=0.65, color="#059669", label="Retained")
    ax.hist(y_proba[y_test==1], bins=40, alpha=0.65, color="#DC2626", label="Churned")
    ax.axvline(0.5, color="#555", ls="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Customers")
    ax.set_title("Churn Risk Score Distribution", pad=12)
    ax.legend()
    save("M1_score_distribution.png")

    # ── Business output: high risk customers ──
    df_out = df.copy()
    X_all  = df_out[FEATURES].values
    df_out["churn_probability"] = model.predict_proba(X_all)[:, 1]
    df_out["churn_risk"]        = pd.cut(
        df_out["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )

    risk_summary = df_out.groupby("churn_risk", observed=True).agg(
        customers=("customer_id","count"),
        avg_monetary=("monetary","mean"),
        avg_recency=("recency_days","mean"),
    ).reset_index()

    print("\n  Churn Risk Segments:")
    print(f"  {'Risk':<8} {'Customers':>10} {'Avg Revenue':>12} {'Avg Recency':>12}")
    print("  " + "-"*46)
    for row in risk_summary.itertuples():
        print(f"  {row.churn_risk:<8} {row.customers:>10,} "
              f"  Rs{row.avg_monetary:>8,.0f}   {row.avg_recency:>8.0f} days")

    at_risk_rev = df_out[df_out["churn_risk"]=="High"]["monetary"].sum()
    print(f"\n  Revenue at risk (High churn probability): Rs{at_risk_rev/1e6:.1f}M")

    with open(os.path.join(REPORT_DIR, "churn_report.txt"), "w") as f:
        f.write(f"Churn Model Report\n{'='*40}\n")
        f.write(f"Test AUC      : {auc:.4f}\n")
        f.write(f"CV AUC (5-fold): {cv.mean():.4f} +/- {cv.std():.4f}\n")
        f.write(f"Revenue at risk: Rs{at_risk_rev/1e6:.1f}M\n\n")
        f.write(classification_report(y_test, y_pred,
                                      target_names=["Retained","Churned"]))

    return model, df_out


# ══════════════════════════════════════════════════════════════
#  MODEL 2 — CUSTOMER SEGMENTATION (K-Means)
# ══════════════════════════════════════════════════════════════

def model_segmentation(df):
    divider("MODEL 2 — Customer Segmentation (K-Means Clustering)")

    SEG_FEATURES = [
        "recency_days","frequency","monetary",
        "avg_order_value","avg_categories_per_order",
        "session_count","avg_session_min","tenure_days",
        "cancel_rate","return_rate",
    ]

    X = df[SEG_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow method to choose K ──
    inertias    = []
    silhouettes = []
    K_range     = range(2, 9)

    print("  Finding optimal K (elbow method)...")
    for k in K_range:
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lbls))
        print(f"    K={k}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(list(K_range), inertias, "o-", color="#2563EB", lw=2.5, ms=8)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Method")
    ax2.plot(list(K_range), silhouettes, "o-", color="#059669", lw=2.5, ms=8)
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Scores")
    plt.suptitle("K-Means: Optimal Cluster Selection", fontsize=14)
    plt.tight_layout()
    save("M2_elbow_silhouette.png")

    # ── Fit with K=4 ──
    K = 4
    km     = KMeans(n_clusters=K, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    print(f"\n  Chosen K=4  Silhouette={sil:.3f}")

    df_seg = df.copy()
    df_seg["cluster"] = labels

    # ── Cluster profiles ──
    profile = df_seg.groupby("cluster")[SEG_FEATURES].mean().round(1)
    profile["size"] = df_seg.groupby("cluster").size()
    profile["monetary_mean"] = df_seg.groupby("cluster")["monetary"].mean().round(0)

    # Name clusters by their dominant traits
    cluster_names = {}
    for cid, row in profile.iterrows():
        if row["recency_days"] < 60 and row["frequency"] > 5:
            cluster_names[cid] = "Champions"
        elif row["monetary_mean"] > profile["monetary_mean"].median() and row["recency_days"] < 90:
            cluster_names[cid] = "Loyal Buyers"
        elif row["recency_days"] > 120:
            cluster_names[cid] = "Hibernating"
        else:
            cluster_names[cid] = "Promising"

    df_seg["cluster_name"] = df_seg["cluster"].map(cluster_names)

    print("\n  Cluster Profiles:")
    print(f"  {'Cluster':<14} {'Size':>6} {'Recency':>9} {'Frequency':>10} "
          f"{'Avg Rev':>10} {'Monetary':>10}")
    print("  " + "-"*62)
    for cid, row in profile.iterrows():
        print(f"  {cluster_names[cid]:<14} {int(row['size']):>6,} "
              f"  {float(row['recency_days']):>7.0f}d  {float(row['frequency']):>9.1f}  "
              f"  Rs{float(row['avg_order_value']):>6,.0f}  Rs{float(row['monetary_mean']):>7,.0f}")

    # ── PCA 2D scatter ──
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in range(K):
        mask = labels == cid
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=18, alpha=0.55, color=PALETTE[cid],
                   label=f"Cluster {cid}: {cluster_names[cid]}")
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    ax.set_title(f"Customer Segments — PCA Projection (K={K}, Silhouette={sil:.3f})", pad=12)
    ax.legend(fontsize=10, markerscale=2)
    save("M2_pca_clusters.png")

    # ── Radar chart: cluster feature averages ──
    features_radar = ["recency_days","frequency","monetary",
                      "avg_order_value","session_count","tenure_days"]
    labels_radar   = ["Recency","Frequency","Monetary","AOV","Sessions","Tenure"]
    N = len(features_radar)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for cid, row in profile.iterrows():
        vals = [(row[f] - profile[f].min()) / (profile[f].max() - profile[f].min() + 1e-9)
                for f in features_radar]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, color=PALETTE[cid],
                label=f"{cluster_names[cid]}")
        ax.fill(angles, vals, alpha=0.08, color=PALETTE[cid])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Cluster Feature Profiles (Normalised)", pad=18, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    save("M2_radar_chart.png")

    with open(os.path.join(REPORT_DIR, "segmentation_report.txt"), "w") as f:
        f.write(f"Segmentation Report\n{'='*40}\n")
        f.write(f"K=4  Silhouette={sil:.3f}\n\n")
        f.write(profile.to_string())

    return df_seg


# ══════════════════════════════════════════════════════════════
#  MODEL 3 — SALES FORECASTING (trend + seasonal decomposition)
# ══════════════════════════════════════════════════════════════

def model_forecast():
    divider("MODEL 3 — Sales Forecasting (Trend + Seasonality)")

    conn = sqlite3.connect(DB_PATH)
    daily = pd.read_sql("""
        SELECT DATE(order_date) AS ds,
               SUM(total_amount) AS y
        FROM orders
        WHERE status = 'Delivered'
        GROUP BY ds
        ORDER BY ds
    """, conn)
    conn.close()

    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.set_index("ds").asfreq("D").fillna(0)

    # ── Weekly & monthly aggregation ──
    weekly  = daily.resample("W")["y"].sum()
    monthly = daily.resample("ME")["y"].sum()

    # ── Build feature matrix for linear model ──
    monthly_df               = monthly.reset_index()
    monthly_df.columns       = ["ds","y"]
    monthly_df["t"]          = np.arange(len(monthly_df))
    monthly_df["month"]      = monthly_df["ds"].dt.month
    monthly_df["is_festive"] = monthly_df["month"].isin([10,11,12]).astype(int)
    monthly_df["is_summer"]  = monthly_df["month"].isin([4,5]).astype(int)
    monthly_df["sin_12"]     = np.sin(2 * np.pi * monthly_df["t"] / 12)
    monthly_df["cos_12"]     = np.cos(2 * np.pi * monthly_df["t"] / 12)

    from sklearn.linear_model import Ridge
    from sklearn.metrics       import mean_absolute_error, mean_absolute_percentage_error

    # Train on first 15 months, forecast last 3
    TRAIN_END = 15
    train = monthly_df.iloc[:TRAIN_END]
    test  = monthly_df.iloc[TRAIN_END:]

    FEAT = ["t","sin_12","cos_12","is_festive","is_summer"]
    reg  = Ridge(alpha=1.0)
    reg.fit(train[FEAT], train["y"])

    train_pred = reg.predict(train[FEAT])
    test_pred  = reg.predict(test[FEAT])

    mae  = mean_absolute_error(test["y"], test_pred)
    mape = mean_absolute_percentage_error(test["y"], test_pred) * 100

    print(f"  Training months : {TRAIN_END}")
    print(f"  Test months     : {len(test)}")
    print(f"  Test MAE        : Rs{mae:,.0f}")
    print(f"  Test MAPE       : {mape:.1f}%")

    # ── Forecast next 6 months ──
    last_t   = monthly_df["t"].max()
    last_ds  = monthly_df["ds"].max()
    future_rows = []
    for i in range(1, 7):
        t_val = last_t + i
        ds    = last_ds + pd.DateOffset(months=i)
        m     = ds.month
        future_rows.append({
            "ds": ds, "t": t_val,
            "sin_12": np.sin(2*np.pi*t_val/12),
            "cos_12": np.cos(2*np.pi*t_val/12),
            "is_festive": int(m in [10,11,12]),
            "is_summer":  int(m in [4,5]),
        })
    future_df = pd.DataFrame(future_rows)
    future_df["forecast"] = reg.predict(future_df[FEAT])

    # simple confidence interval via training residuals
    residuals = train["y"].values - train_pred
    std_res   = residuals.std()
    future_df["lower"] = future_df["forecast"] - 1.64 * std_res
    future_df["upper"] = future_df["forecast"] + 1.64 * std_res

    print("\n  6-Month Revenue Forecast:")
    print(f"  {'Month':<12} {'Forecast':>12} {'Lower (90%)':>13} {'Upper (90%)':>13}")
    print("  " + "-"*52)
    for row in future_df.itertuples():
        print(f"  {str(row.ds)[:7]:<12}  Rs{row.forecast/1e6:>6.1f}M"
              f"    Rs{row.lower/1e6:>6.1f}M    Rs{row.upper/1e6:>6.1f}M")

    # ── Forecast chart ──
    fig, ax = plt.subplots(figsize=(13, 5))

    # historical
    ax.plot(monthly_df["ds"], monthly_df["y"]/1e6,
            color="#2563EB", lw=2, label="Historical")

    # in-sample fit
    ax.plot(train["ds"], train_pred/1e6,
            color="#059669", lw=1.5, ls="--", alpha=0.8, label="Model fit")

    # test actual vs predicted
    ax.plot(test["ds"], test["y"]/1e6,
            color="#2563EB", lw=2)
    ax.plot(test["ds"], test_pred/1e6,
            color="#DC2626", lw=2, ls="--", label=f"Test predictions (MAPE {mape:.1f}%)")

    # future forecast + CI
    ax.plot(future_df["ds"], future_df["forecast"]/1e6,
            color="#D97706", lw=2.5, ls="-", marker="o", ms=6, label="Forecast (6M)")
    ax.fill_between(future_df["ds"],
                    future_df["lower"]/1e6, future_df["upper"]/1e6,
                    alpha=0.18, color="#D97706", label="90% CI")

    # dividing line
    ax.axvline(test["ds"].min(), color="#888", ls=":", lw=1.5, label="Train/Test split")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"Rs{x:.0f}M"))
    ax.set_xlabel("")
    ax.set_ylabel("Monthly Revenue")
    ax.set_title("Sales Forecast — Historical + 6-Month Outlook", fontsize=14, pad=12)
    ax.legend(fontsize=9, loc="upper left")
    save("M3_sales_forecast.png")

    # ── Weekly decomposition chart ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    ax1.plot(weekly.index, weekly.values/1e3, color="#2563EB", lw=1.5)
    ax1.set_ylabel("Revenue (Rs K)")
    ax1.set_title("Weekly Revenue", fontsize=12)

    rolling_avg = weekly.rolling(4, center=True).mean()
    ax1.plot(rolling_avg.index, rolling_avg.values/1e3,
             color="#DC2626", lw=2.5, label="4-week rolling avg")
    ax1.legend(fontsize=9)

    monthly_df["residual"] = monthly_df["y"] - reg.predict(monthly_df[FEAT])
    ax2.bar(monthly_df["ds"], monthly_df["residual"]/1e6,
            color=["#059669" if v>=0 else "#DC2626" for v in monthly_df["residual"]],
            width=20)
    ax2.axhline(0, color="#888", lw=1)
    ax2.set_ylabel("Residual (Rs M)")
    ax2.set_title("Model Residuals by Month", fontsize=12)

    plt.suptitle("Revenue Time-Series Decomposition", fontsize=14, y=1.01)
    plt.tight_layout()
    save("M3_decomposition.png")

    with open(os.path.join(REPORT_DIR, "forecast_report.txt"), "w") as f:
        f.write(f"Forecast Report\n{'='*40}\n")
        f.write(f"Test MAE  : Rs{mae:,.0f}\n")
        f.write(f"Test MAPE : {mape:.1f}%\n\n")
        f.write("6-Month Forecast:\n")
        f.write(future_df[["ds","forecast","lower","upper"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("RetailPulse — Phase 4: Machine Learning\n")

    print("Building customer feature matrix...")
    df = build_customer_features()
    print(f"  Feature matrix: {df.shape[0]:,} customers x {df.shape[1]} features\n")

    churn_model, df_with_scores = model_churn(df)
    df_segmented                = model_segmentation(df)
    model_forecast()

    print("\n" + "═"*60)
    print("  Phase 4 complete!")
    print("  Charts  → models/charts/")
    print("  Reports → models/reports/")
    print("\n  Models built:")
    print("    M1  Churn Prediction    (Random Forest — AUC check report)")
    print("    M2  Segmentation        (K-Means, K=4, PCA visualised)")
    print("    M3  Sales Forecast      (Ridge regression + seasonality)")
    print("\n  Next → Phase 5: Streamlit Dashboard  (dashboard/app.py)")
    print("═"*60)
