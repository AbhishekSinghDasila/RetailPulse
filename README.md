## RetailPulse – Sales Analytics & Business Intelligence Dashboard

# Overview

RetailPulse is an end-to-end Sales Analytics and Business Intelligence project built using Python, SQL, Streamlit, and Machine Learning.

The project simulates a real-world retail business where thousands of customer transactions are analyzed to generate actionable business insights.

Instead of simply visualizing data, RetailPulse follows a complete analytics workflow:

Data Collection
Data Cleaning
Data Transformation
Exploratory Data Analysis (EDA)
KPI Generation
Interactive Dashboard Development
Sales Forecasting
Business Recommendations

The goal is to help business stakeholders answer important questions such as:

Which products generate the highest revenue?
Which cities perform the best?
What are the monthly sales trends?
Who are the most valuable customers?
Which product categories need attention?
How can future sales be predicted?
Project Objectives

The main objectives of this project are:

Analyze retail sales data.
Build interactive business dashboards.
Generate KPIs for management.
Discover hidden business patterns.
Forecast future sales using Machine Learning.
Demonstrate practical Data Analysis skills required in industry.
Tech Stack
Category	Technologies
Programming	Python
Dashboard	Streamlit
Database	MySQL
Data Analysis	Pandas, NumPy
Visualization	Plotly, Matplotlib, Seaborn
Machine Learning	Scikit-Learn
Version Control	Git & GitHub

## Project Structure
RetailPulse/
```
│
├── dashboard/
│     ├── app.py
│     ├── pages/
│     └── assets/
│
├── data/
│     ├── customers.csv
│     ├── products.csv
│     ├── orders.csv
│     ├── order_items.csv
│     └── stores.csv
│
├── notebooks/
│     ├── data_cleaning.ipynb
│     ├── eda.ipynb
│     └── forecasting.ipynb
│
├── models/
│     └── sales_forecast.pkl
│
├── requirements.txt
└── README.md

```

## Dataset Description

The project contains multiple relational datasets similar to an actual retail company's database.

Customers

Contains customer information.

Columns include:
```
Customer ID
Name
Gender
Age
City
State
Products
```

Contains product details.

Columns include:
```
Product ID
Product Name
Category
Cost Price
Selling Price
Orders
```

Contains order level information.

Columns include:
```
Order ID
Customer ID
Order Date
Store ID
Order Items
```

Contains transaction details.

Columns include:

Order ID
Product ID
Quantity
Unit Price
Stores

Contains store information.

Columns include:

Store ID
Store Name
City
Region
Data Analysis Workflow

The project follows a complete analytics pipeline.

Step 1 – Data Loading
Load CSV files
Merge multiple tables
Verify schema
Handle missing values


## Step 2 – Data Cleaning

Performed:

Duplicate removal
Missing value handling
Data type conversion
Invalid value correction
Date formatting

## Step 3 – Feature Engineering

Created useful business features such as:

Revenue
Profit
Month
Quarter
Year
Profit Margin
Average Order Value

## Step 4 – Exploratory Data Analysis

Performed detailed EDA to answer business questions.

Examples:

Monthly Sales Trend
Revenue by City
Revenue by Product
Top Customers
Category Performance
Order Distribution
Seasonal Trends

## Step 5 – KPI Generation

Important KPIs displayed in dashboard.

Examples:

Total Revenue
Total Profit
Total Orders
Average Order Value
Number of Customers
Best Selling Product
Best Performing City

## Step 6 – Dashboard Development

Developed an interactive Streamlit dashboard with filters.

Features include:

Date Filters
Category Filters
City Filters
Product Filters
KPI Cards
Interactive Charts
Drill-down Analysis

## Step 7 – Sales Forecasting

Machine Learning model predicts future sales trends.

Typical workflow:

Historical sales aggregation
Feature creation
Train-test split
Model training
Prediction
Performance evaluation
Dashboard Features

The dashboard contains multiple business insights.

Executive Summary

Shows:

Total Revenue
Total Orders
Total Customers
Total Profit
Sales Analysis

Displays:

Daily Sales
Monthly Sales
Yearly Sales
Sales Growth
Customer Analysis

Provides insights into:

Top Customers
Customer Distribution
Repeat Customers
Average Spending
Product Analysis

Shows:

Top Selling Products
Least Selling Products
Category Revenue
Product Profitability
Regional Analysis

Displays:

Revenue by City
Revenue by Region
Store Performance
Forecasting

Predicts future sales based on historical trends.

Visualizations Used

The project includes:

Line Charts
Bar Charts
Pie Charts
Donut Charts
Scatter Plots
Heatmaps
Histograms
Box Plots
Key Business Insights

Some examples of insights generated include:

Revenue is concentrated among a few top-performing products.
Certain cities contribute significantly more sales than others.
Seasonal demand causes noticeable spikes in revenue.
A small percentage of customers contribute a large portion of total sales.
Some categories generate high sales but relatively low profit margins.
Skills Demonstrated

This project demonstrates practical knowledge of:
```
Python
Functions
Modules
File Handling
OOP Concepts
Pandas
Data Cleaning
Merge
GroupBy
Pivot Tables
Aggregations
Feature Engineering
NumPy
Numerical Operations
Array Manipulation
SQL
Joins
Group By
Aggregations
Window Functions
Analytical Queries
Data Visualization
Plotly
Matplotlib
Seaborn
Interactive Charts
Streamlit
Dashboard Development
Sidebar Filters
KPI Cards
Interactive Components
Machine Learning
Data Preprocessing
Feature Selection
Model Training
Prediction
Evaluation
Installation
```

Clone the repository

git clone https://github.com/AbhishekSinghDasila/RetailPulse.git

Move into the project

cd RetailPulse

Install dependencies

pip install -r requirements.txt

Run the dashboard

streamlit run dashboard/app.py
Future Improvements

Possible enhancements include:

Customer Segmentation using K-Means
```
Recommendation System
Inventory Forecasting
Time Series Forecasting using LSTM
Real-time Dashboard
Cloud Deployment
Power BI Integration
Docker Support
CI/CD Pipeline
Interview Questions You Should Be Ready To Answer
```

Why Choose Streamlit
```
Because it allows rapid development of interactive dashboards directly in Python without requiring frontend technologies.
```

Why use Plotly instead of Matplotlib?
```
Plotly provides interactive visualizations such as zooming, hovering, filtering, and dynamic legends, making dashboards more user-friendly.
```

Why use Pandas?
```
Pandas simplifies data cleaning, transformation, aggregation, and analysis of structured datasets.
```

What KPIs did you calculate?
```
Examples include:

Total Revenue
Profit
Average Order Value
Monthly Growth
Customer Count
Product Sales
Profit Margin
What business value does this project provide?
```

The dashboard enables management to:

Monitor business performance
Identify profitable products
Track customer behavior
Optimize inventory planning
Forecast future sales
Support data-driven decision making
Learning Outcomes

Through this project, I gained practical experience in:

Data Cleaning
Exploratory Data Analysis
Dashboard Development
SQL Querying
Business Intelligence
Data Visualization
Machine Learning
Project Deployment
End-to-End Analytics Workflow
