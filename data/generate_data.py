import sqlite3
import random
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

DB_PATH     = os.path.join(os.path.dirname(__file__), "retailpulse.db")
N_CUSTOMERS = 2_000
N_PRODUCTS  = 500
N_ORDERS    = 52_000
N_SESSIONS  = 120_000
START_DATE  = datetime(2023, 1, 1)
END_DATE    = datetime(2024, 6, 30)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rdate(start=START_DATE, end=END_DATE):
    delta = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, delta))

def wchoice(opts, wts):
    return random.choices(opts, weights=wts, k=1)[0]

def rand_name():
    FIRST = ["Aarav","Aditya","Amit","Ananya","Anjali","Arjun","Deepak","Divya",
             "Gaurav","Ishaan","Kavya","Kiran","Meera","Neeraj","Neha","Pooja",
             "Priya","Rahul","Rajesh","Riya","Rohit","Sakshi","Sanjay","Shruti",
             "Sneha","Suresh","Tanvi","Varun","Vikram","Zara"]
    LAST  = ["Sharma","Verma","Patel","Singh","Kumar","Gupta","Joshi","Mehta",
             "Shah","Rao","Nair","Iyer","Reddy","Chopra","Malhotra","Bose",
             "Das","Chatterjee","Mishra","Sinha"]
    return f"{random.choice(FIRST)} {random.choice(LAST)}"

def rand_email(name, i):
    domains = ["gmail.com","yahoo.com","hotmail.com","outlook.com","rediffmail.com"]
    slug = name.lower().replace(" ",".")[:12] + str(i)
    return f"{slug}@{random.choice(domains)}"


# ── 1. Customers ──────────────────────────────────────────────────────────────

CITIES   = ["Mumbai","Delhi","Bangalore","Hyderabad","Chennai",
            "Kolkata","Pune","Ahmedabad","Jaipur","Lucknow"]
C_WGTS   = [18,16,14,10,10,8,8,7,5,4]
SEGMENTS = ["New","Loyal","At-Risk","Champions","Hibernating"]
SEG_WGTS = [30,25,20,15,10]
CHANNELS = ["Organic","Paid Search","Social","Email","Referral","Direct"]
CH_WGTS  = [28,22,18,15,10,7]


def generate_customers() -> pd.DataFrame:
    print("  [1/5] Customers...")
    rows = []
    for i in range(1, N_CUSTOMERS+1):
        name   = rand_name()
        signup = rdate(START_DATE, END_DATE - timedelta(days=30))
        rows.append({
            "customer_id":          i,
            "name":                 name,
            "email":                rand_email(name, i),
            "city":                 wchoice(CITIES, C_WGTS),
            "age":                  int(np.clip(np.random.normal(34,10), 18, 70)),
            "gender":               wchoice(["M","F","Other"], [52,45,3]),
            "segment":              wchoice(SEGMENTS, SEG_WGTS),
            "acquisition_channel":  wchoice(CHANNELS, CH_WGTS),
            "signup_date":          signup.date().isoformat(),
            "is_active":            wchoice([1,0], [75,25]),
        })
    return pd.DataFrame(rows)


# ── 2. Products ───────────────────────────────────────────────────────────────

CATEGORIES = {
    "Electronics":    (20, 5_000,  80_000),
    "Clothing":       (18,   299,   5_000),
    "Home & Kitchen": (15,   499,  15_000),
    "Books":          (12,    99,   1_500),
    "Beauty":         (10,   199,   4_000),
    "Sports":         (10,   499,  12_000),
    "Toys":           ( 8,   299,   3_500),
    "Grocery":        ( 7,    50,   1_200),
}
ADJ   = ["Premium","Classic","Ultra","Pro","Lite","Smart","Eco","Slim","Rapid","Cozy"]
NOUNS = {
    "Electronics":    ["Headphones","Charger","Keyboard","Mouse","Webcam","Tablet","Speaker","Hub"],
    "Clothing":       ["T-Shirt","Jeans","Jacket","Dress","Kurta","Sneakers","Hoodie","Cap"],
    "Home & Kitchen": ["Pan","Blender","Curtains","Lamp","Pillow","Organiser","Kettle","Spatula"],
    "Books":          ["Novel","Textbook","Guide","Journal","Comic","Planner","Atlas","Diary"],
    "Beauty":         ["Serum","Moisturiser","Lipstick","Cleanser","Toner","Mask","Sunscreen","Oil"],
    "Sports":         ["Dumbbell","Mat","Gloves","Bottle","Band","Rope","Helmet","Pad"],
    "Toys":           ["Puzzle","Car","Doll","LEGO Set","Board Game","Playset","Slime","Robot"],
    "Grocery":        ["Almonds","Oats","Honey","Coffee","Protein Bar","Spice Mix","Tea","Ghee"],
}


def generate_products() -> pd.DataFrame:
    print("  [2/5] Products...")
    cats  = list(CATEGORIES.keys())
    cwgts = [CATEGORIES[c][0] for c in cats]
    rows  = []
    for pid in range(1, N_PRODUCTS+1):
        cat = wchoice(cats, cwgts)
        _, lo, hi = CATEGORIES[cat]
        price = round(random.uniform(lo, hi), -1)
        rows.append({
            "product_id":   pid,
            "name":         f"{random.choice(ADJ)} {random.choice(NOUNS[cat])}",
            "category":     cat,
            "price":        price,
            "cost":         round(price * random.uniform(0.35,0.65), 2),
            "stock":        random.randint(0, 500),
            "rating":       round(random.uniform(3.0, 5.0), 1),
            "review_count": random.randint(0, 5_000),
            "is_active":    wchoice([1,0], [88,12]),
        })
    return pd.DataFrame(rows)


# ── 3. Orders & Order Items ───────────────────────────────────────────────────

STATUSES = ["Delivered","Shipped","Processing","Cancelled","Returned"]
S_WGTS   = [65,12,8,10,5]
PAYMENTS = ["UPI","Credit Card","Debit Card","Net Banking","COD","Wallet"]
P_WGTS   = [35,22,15,10,12,6]


def generate_orders_and_items(customers, products):
    print("  [3/5] Orders & order items (takes ~10s)...")
    custs = customers.to_dict("records")
    prods = products[products["is_active"]==1].to_dict("records")
    orders, items = [], []
    oid = iid = 1

    for _ in range(N_ORDERS):
        c      = random.choice(custs)
        signup = datetime.fromisoformat(c["signup_date"])
        odate  = rdate(max(signup + timedelta(days=1), START_DATE), END_DATE)
        status = wchoice(STATUSES, S_WGTS)
        n_items= random.choices([1,2,3,4,5], weights=[45,28,15,8,4])[0]
        chosen = random.sample(prods, min(n_items, len(prods)))

        subtotal = 0.0
        for p in chosen:
            qty  = random.choices([1,2,3], weights=[70,22,8])[0]
            disc = random.choices([0,.05,.10,.20,.30], weights=[40,20,20,12,8])[0]
            lt   = round(p["price"] * qty * (1-disc), 2)
            subtotal += lt
            items.append({
                "item_id":     iid,
                "order_id":    oid,
                "product_id":  p["product_id"],
                "quantity":    qty,
                "unit_price":  p["price"],
                "discount_pct":disc,
                "line_total":  lt,
            })
            iid += 1

        shipping = 0 if subtotal > 499 else 49
        tax      = round(subtotal * 0.18, 2)
        del_at   = None
        if status == "Delivered":
            del_at = (odate + timedelta(days=random.randint(2,7))).isoformat()

        orders.append({
            "order_id":       oid,
            "customer_id":    c["customer_id"],
            "order_date":     odate.isoformat(),
            "status":         status,
            "payment_method": wchoice(PAYMENTS, P_WGTS),
            "subtotal":       round(subtotal, 2),
            "shipping_fee":   shipping,
            "tax":            tax,
            "total_amount":   round(subtotal + shipping + tax, 2),
            "delivered_at":   del_at,
        })
        oid += 1

    return pd.DataFrame(orders), pd.DataFrame(items)


# ── 4. Sessions (web funnel) ──────────────────────────────────────────────────

PAGES   = ["home","category","product","cart","checkout","confirmation"]
DEVICES = ["mobile","desktop","tablet"]
D_WGTS  = [55,38,7]


def generate_sessions(customers):
    print("  [4/5] Web sessions...")
    cids = customers["customer_id"].tolist()
    rows = []
    for i in range(1, N_SESSIONS+1):
        anon  = random.random() < 0.30
        depth = random.choices(range(1,7), weights=[30,25,20,12,8,5])[0]
        rows.append({
            "session_id":       i,
            "customer_id":      None if anon else random.choice(cids),
            "session_start":    rdate().isoformat(),
            "duration_seconds": random.randint(30, 1_800),
            "device":           wchoice(DEVICES, D_WGTS),
            "channel":          wchoice(CHANNELS, CH_WGTS),
            "landing_page":     PAGES[0],
            "exit_page":        PAGES[depth-1],
            "pages_viewed":     depth,
            "converted":        int(depth == 6),
        })
    return pd.DataFrame(rows)


# ── 5. Load to SQLite ─────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id         INTEGER PRIMARY KEY,
    name                TEXT    NOT NULL,
    email               TEXT    UNIQUE NOT NULL,
    city                TEXT,
    age                 INTEGER,
    gender              TEXT,
    segment             TEXT,
    acquisition_channel TEXT,
    signup_date         DATE,
    is_active           INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    price        REAL NOT NULL,
    cost         REAL,
    stock        INTEGER DEFAULT 0,
    rating       REAL,
    review_count INTEGER DEFAULT 0,
    is_active    INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS orders (
    order_id       INTEGER PRIMARY KEY,
    customer_id    INTEGER NOT NULL,
    order_date     DATETIME NOT NULL,
    status         TEXT NOT NULL,
    payment_method TEXT,
    subtotal       REAL,
    shipping_fee   REAL DEFAULT 0,
    tax            REAL DEFAULT 0,
    total_amount   REAL NOT NULL,
    delivered_at   DATETIME,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE TABLE IF NOT EXISTS order_items (
    item_id      INTEGER PRIMARY KEY,
    order_id     INTEGER NOT NULL,
    product_id   INTEGER NOT NULL,
    quantity     INTEGER NOT NULL DEFAULT 1,
    unit_price   REAL NOT NULL,
    discount_pct REAL DEFAULT 0,
    line_total   REAL NOT NULL,
    FOREIGN KEY (order_id)   REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id       INTEGER PRIMARY KEY,
    customer_id      INTEGER,
    session_start    DATETIME NOT NULL,
    duration_seconds INTEGER,
    device           TEXT,
    channel          TEXT,
    landing_page     TEXT,
    exit_page        TEXT,
    pages_viewed     INTEGER DEFAULT 1,
    converted        INTEGER DEFAULT 0,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_customer   ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date       ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders(status);
CREATE INDEX IF NOT EXISTS idx_items_order       ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_items_product     ON order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_sessions_customer ON sessions(customer_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date     ON sessions(session_start);
"""


def load_to_db(customers, products, orders, items, sessions):
    print(f"  [5/5] Writing to SQLite → {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    customers.to_sql("customers",   conn, if_exists="replace", index=False)
    products.to_sql("products",     conn, if_exists="replace", index=False)
    orders.to_sql("orders",         conn, if_exists="replace", index=False)
    items.to_sql("order_items",     conn, if_exists="replace", index=False)
    sessions.to_sql("sessions",     conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


# ── 6. Validation ─────────────────────────────────────────────────────────────

def validate_db():
    conn = sqlite3.connect(DB_PATH)
    print("\n  Table             Rows")
    print("  " + "-"*26)
    for t in ["customers","products","orders","order_items","sessions"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<18} {n:>8,}")

    rev = conn.execute(
        "SELECT ROUND(SUM(total_amount)/1e6,2) FROM orders WHERE status='Delivered'"
    ).fetchone()[0]
    cat = conn.execute("""
        SELECT p.category, ROUND(SUM(oi.line_total)/1e6,2) rev
        FROM order_items oi JOIN products p ON oi.product_id=p.product_id
        GROUP BY p.category ORDER BY rev DESC LIMIT 1
    """).fetchone()
    conv = conn.execute(
        "SELECT ROUND(100.0*SUM(converted)/COUNT(*),1) FROM sessions"
    ).fetchone()[0]
    conn.close()
    print(f"\n  Total delivered revenue : Rs {rev}M")
    print(f"  Top revenue category    : {cat[0]} (Rs {cat[1]}M)")
    print(f"  Session conversion rate : {conv}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RetailPulse — Phase 1: Generating synthetic e-commerce data\n")
    customers        = generate_customers()
    products         = generate_products()
    orders, items    = generate_orders_and_items(customers, products)
    sessions         = generate_sessions(customers)
    load_to_db(customers, products, orders, items, sessions)
    validate_db()
    print("\n  Done! -> data/retailpulse.db")
