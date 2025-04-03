# Import required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import calendar
import random
from datetime import datetime

# Set display options for better visualization
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Set random seed for reproducibility
np.random.seed(datetime.now().microsecond)

# Load the datasets
flight_activity = pd.read_csv('dataset/Customer Flight Activity.csv')
loyalty_history = pd.read_csv('dataset/Customer Loyalty History.csv')

# Enhanced Data Preprocessing with available columns
def preprocess_data():
    # Merge the datasets
    df = pd.merge(flight_activity, loyalty_history, on='Loyalty Number', how='inner')
    
    # Calculate average ticket price using Dollar Cost Points Redeemed and Total Flights
    df['Avg_Ticket_Price'] = df['Dollar Cost Points Redeemed'] / df['Total Flights']
    df['Avg_Ticket_Price'] = df['Avg_Ticket_Price'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate ancillary revenue (approximation)
    df['Ancillary'] = df['Points Redeemed'] * 0.02  # Assuming $0.02 value per point
    
    # Calculate FM metrics (Frequency and Monetary)
    frequency = df.groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    monetary = df.groupby('Loyalty Number')['CLV'].first().reset_index()
    ancillary = df.groupby('Loyalty Number')['Ancillary'].sum().reset_index()
    avg_price = df.groupby('Loyalty Number')['Avg_Ticket_Price'].mean().reset_index()
    
    # Analyze temporal patterns
    df['Year_Month'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    monthly_flights = df.groupby('Year_Month')['Total Flights'].sum().reset_index()
    peak_months = monthly_flights.nlargest(3, 'Total Flights')['Year_Month'].dt.month.tolist()
    
    # Calculate seasonal patterns
    df['Season'] = pd.cut(df['Month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'],
                         include_lowest=True)
    seasonal_stats = df.groupby('Season', observed=True).agg({
        'Total Flights': 'sum',
        'CLV': 'mean',
        'Points Accumulated': 'sum',
        'Dollar Cost Points Redeemed': 'sum'
    }).reset_index()
    
    # Combine metrics
    fm = pd.merge(frequency, monetary, on='Loyalty Number')
    fm = pd.merge(fm, ancillary, on='Loyalty Number')
    fm = pd.merge(fm, avg_price, on='Loyalty Number')
    fm.columns = ['Loyalty Number', 'Frequency', 'Monetary', 'Ancillary', 'Avg_Ticket_Price']
    
    # Add loyalty card type and other relevant features
    loyalty_card = df.groupby('Loyalty Number')['Loyalty Card'].first().reset_index()
    fm = pd.merge(fm, loyalty_card, on='Loyalty Number')
    
    # Calculate loyalty tier statistics
    loyalty_tier_stats = fm.groupby('Loyalty Card').agg({
        'Loyalty Number': 'count',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Avg_Ticket_Price': 'mean',
        'Ancillary': 'mean'
    }).reset_index()
    loyalty_tier_stats['Percentage'] = (loyalty_tier_stats['Loyalty Number'] / loyalty_tier_stats['Loyalty Number'].sum()) * 100
    
    return fm, df, seasonal_stats, peak_months, loyalty_tier_stats

# Enhanced Business Analysis with new metrics
def analyze_business_cases(fm, df):
    # Feature Scaling
    scaler = StandardScaler()
    fm_scaled = scaler.fit_transform(fm[['Frequency', 'Monetary']])
    fm_scaled = pd.DataFrame(fm_scaled, columns=['Frequency', 'Monetary'])
    
    # Apply K-means clustering with explicit n_init to avoid warning
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    fm['Cluster'] = kmeans.fit_predict(fm_scaled)
    
    # Calculate new business metrics
    customer_revenue = df.groupby('Loyalty Number').agg({
        'Dollar Cost Points Redeemed': 'sum',
        'Ancillary': 'sum'
    }).reset_index()
    customer_revenue['Total_Revenue'] = customer_revenue['Dollar Cost Points Redeemed'] + customer_revenue['Ancillary']
    fm = pd.merge(fm, customer_revenue[['Loyalty Number', 'Total_Revenue']], on='Loyalty Number')
    
    # Calculate churn risk based on activity
    fm['Churn_Risk'] = np.where(
        (fm['Frequency'] < fm['Frequency'].median()) & (fm['Monetary'] < fm['Monetary'].median()),
        'High',
        np.where(
            (fm['Frequency'] < fm['Frequency'].median()) | (fm['Monetary'] < fm['Monetary'].median()),
            'Medium',
            'Low'
        )
    )
    
    return fm

# Dynamic Recommendation Generator
def generate_dynamic_recommendations(cluster_stats, peak_season, peak_months):
    # Base recommendation templates
    base_recommendations = {
        'Cluster 0': {
            'title': 'High Frequency, Moderate Value Customers',
            'base': [
                "Implement dynamic pricing for frequent flyers",
                "Offer volume discounts for multiple bookings",
                "Create bundled packages with partner services",
                "Develop corporate travel programs",
                "Introduce family/group travel incentives"
            ],
            'seasonal': {
                'Summer': ["Summer travel passes", "Early bird specials", "Family vacation bundles"],
                'Winter': ["Holiday season packages", "Ski destination deals", "Winter getaway promotions"],
                'Spring': ["Spring break offers", "Destination discovery deals", "Shoulder season discounts"],
                'Fall': ["Fall foliage tours", "Post-summer discounts", "Cultural experience packages"]
            },
            'upsell': [
                "Priority boarding upgrades",
                "Extra baggage allowances",
                "Lounge access passes",
                "Premium meal options",
                "Seat selection fees"
            ]
        },
        'Cluster 1': {
            'title': 'Low Frequency, Moderate Value Customers',
            'base': [
                "Create seasonal promotions",
                "Offer limited-time discounts",
                "Develop weekend getaway packages",
                "Implement flexible booking options",
                "Provide sign-up bonuses"
            ],
            'seasonal': {
                'Summer': ["Summer vacation deals", "Beach destination packages", "Adventure travel offers"],
                'Winter': ["Winter sun specials", "Festival travel packages", "New Year getaway deals"],
                'Spring': ["Flower festival tours", "Spring city breaks", "Outdoor activity packages"],
                'Fall': ["Fall color tours", "Wine country visits", "Cultural festival packages"]
            },
            'upsell': [
                "Hotel bundle discounts",
                "Car rental packages",
                "Travel insurance offers",
                "Airport transfer services",
                "Local experience packages"
            ]
        },
        'Cluster 2': {
            'title': 'High Value Customers',
            'base': [
                "Offer premium seat upgrades",
                "Provide personalized concierge services",
                "Create VIP event packages",
                "Develop exclusive destination partnerships",
                "Assign dedicated account managers"
            ],
            'seasonal': {
                'Summer': ["Luxury summer retreats", "Private island getaways", "Yacht charter packages"],
                'Winter': ["Alpine luxury experiences", "Private jet ski trips", "Designer shopping tours"],
                'Spring': ["Gourmet food tours", "Private vineyard visits", "Art and culture experiences"],
                'Fall': ["Luxury train journeys", "Five-star safari packages", "Private cultural tours"]
            },
            'upsell': [
                "First class upgrades",
                "Executive lounge memberships",
                "Chauffeur services",
                "Five-star hotel partnerships",
                "Exclusive event access"
            ]
        },
        'Cluster 3': {
            'title': 'Medium-High Frequency, High Value Customers',
            'base': [
                "Implement dynamic loyalty rewards",
                "Create business travel packages",
                "Offer premium tier benefits",
                "Develop destination-specific programs",
                "Provide flexible corporate solutions"
            ],
            'seasonal': {
                'Summer': ["Business summer deals", "Conference travel packages", "Networking event bundles"],
                'Winter': ["Year-end business retreats", "Corporate holiday packages", "Executive winter getaways"],
                'Spring': ["Industry summit packages", "Business class spring deals", "Corporate training retreats"],
                'Fall': ["Fall business conferences", "Executive leadership retreats", "Corporate strategy offsites"]
            },
            'upsell': [
                "Business class upgrades",
                "Airport fast-track services",
                "Premium workspace access",
                "Dedicated check-in services",
                "Corporate travel management"
            ]
        }
    }

    recommendations = {}
    for cluster in base_recommendations:
        # Randomly select 3-4 base recommendations
        base_recs = random.sample(base_recommendations[cluster]['base'], k=random.randint(3,4))
        
        # Get 1-2 seasonal recommendations
        seasonal_recs = random.sample(base_recommendations[cluster]['seasonal'][peak_season], k=random.randint(1,2))
        
        # Get 2-3 upsell opportunities
        upsell_ops = random.sample(base_recommendations[cluster]['upsell'], k=random.randint(2,3))
        
        # Determine price sensitivity dynamically
        sensitivity_options = ['Low', 'Moderate', 'High']
        weights = {
            'Cluster 0': [0.1, 0.7, 0.2],
            'Cluster 1': [0.05, 0.3, 0.65],
            'Cluster 2': [0.7, 0.25, 0.05],
            'Cluster 3': [0.4, 0.5, 0.1]
        }
        price_sensitivity = random.choices(sensitivity_options, weights=weights[cluster], k=1)[0]
        
        recommendations[cluster] = {
            'title': base_recommendations[cluster]['title'],
            'revenue_contribution': f"{cluster_stats.loc[int(cluster.split()[-1]), 'Revenue_Contribution']:.1f}%",
            'recommendations': base_recs + seasonal_recs,
            'price_sensitivity': price_sensitivity,
            'upsell_opportunities': upsell_ops
        }
    
    return recommendations

# Custom Peach Skyline color palette
peach_skyline_colors = [
    '#FFD3B6',  # Light peach
    '#FFAAA5',  # Soft coral
    '#FF8B94',  # Salmon pink
    '#CC7E85',  # Dusty rose
    '#A37F74',  # Muted terracotta
    '#6B5B95',  # Deep lavender (accent)
]

# Enhanced Dashboard with Peach Skyline theme
def create_dashboard(fm, df, seasonal_stats, peak_months, loyalty_tier_stats):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    
    # Calculate cluster statistics for new metrics
    cluster_stats = fm.groupby('Cluster').agg({
        'Monetary': 'mean',
        'Frequency': 'mean',
        'Total_Revenue': 'sum',
        'Avg_Ticket_Price': 'mean',
        'Ancillary': 'mean',
        'Loyalty Number': 'count'
    }).reset_index()
    cluster_stats['Revenue_Contribution'] = (cluster_stats['Total_Revenue'] / cluster_stats['Total_Revenue'].sum()) * 100
    cluster_stats['Segment_Size'] = (cluster_stats['Loyalty Number'] / cluster_stats['Loyalty Number'].sum()) * 100
    
    # Calculate key metrics for overview
    total_customers = fm['Loyalty Number'].nunique()
    avg_ticket_price = fm['Avg_Ticket_Price'].mean()
    total_revenue = fm['Total_Revenue'].sum()
    peak_season = seasonal_stats.iloc[seasonal_stats['Total Flights'].idxmax()]['Season']
    
    # Generate initial dynamic recommendations
    strategic_recommendations = generate_dynamic_recommendations(cluster_stats, peak_season, peak_months)
    
    # Custom CSS with Peach Skyline theme and serif fonts
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Airline Customer Segmentation Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: Georgia, 'Times New Roman', Times, serif;
                    font-size: 18px;
                    line-height: 1.6;
                    background-color: #222;
                    color: #eee;
                }
                .dashboard-header {
                    background: linear-gradient(135deg, #2c3e50 0%, #6B5B95 100%);
                    color: white;
                    padding: 2rem 0;
                    margin-bottom: 2rem;
                    border-bottom: 1px solid #444;
                }
                .dashboard-header h1 {
                    font-size: 2.5rem;
                    font-weight: 600;
                }
                .dashboard-header p {
                    font-size: 1.25rem;
                }
                .card {
                    font-size: 18px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                    margin-bottom: 1.5rem;
                    background-color: #2c3e50;
                    border: 1px solid #444;
                }
                .card-header {
                    font-size: 1.5rem;
                    font-weight: 500;
                    background: linear-gradient(135deg, #6B5B95 0%, #A37F74 100%);
                    color: white;
                    border-radius: 10px 10px 0 0;
                    border-bottom: 1px solid #444;
                }
                .nav-tabs .nav-link {
                    font-size: 1.1rem;
                    color: #FFD3B6;
                    font-weight: 500;
                    background-color: #2c3e50;
                    border: 1px solid #444;
                }
                .nav-tabs .nav-link.active {
                    font-size: 1.1rem;
                    color: #FF8B94;
                    font-weight: 600;
                    background-color: #2c3e50;
                    border-color: #444 #444 #2c3e50;
                }
                .graph-container {
                    font-size: 18px;
                    background-color: #2c3e50;
                    padding: 1.5rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    border: 1px solid #444;
                }
                .graph-container h3 {
                    font-size: 1.5rem;
                    margin-bottom: 1.5rem;
                }
                .graph-container h4 {
                    font-size: 1.3rem;
                    margin-bottom: 1.2rem;
                }
                .metric-card {
                    font-size: 18px;
                    background: linear-gradient(135deg, #A37F74 0%, #6B5B95 100%);
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                    border: 1px solid #444;
                }
                .metric-title {
                    font-size: 1rem;
                    color: #FFD3B6;
                    margin-bottom: 0.5rem;
                }
                .metric-value {
                    font-size: 1.8rem;
                    color: white;
                    font-weight: bold;
                }
                .dropdown-menu {
                    font-size: 18px;
                    background-color: #2c3e50;
                    border: 1px solid #444;
                }
                .dropdown-item {
                    font-size: 18px;
                    color: #FFD3B6;
                    background-color: #2c3e50;
                }
                .dropdown-item:hover {
                    font-size: 18px;
                    background-color: #6B5B95;
                    color: white;
                }
                .VirtualizedSelectOption {
                    font-size: 18px;
                    color: #FFD3B6 !important;
                    background-color: #2c3e50 !important;
                }
                .VirtualizedSelectFocusedOption {
                    background-color: #6B5B95 !important;
                    color: white !important;
                }
                .Select-menu-outer {
                    background-color: #2c3e50 !important;
                    border: 1px solid #444 !important;
                }
                .Select-control {
                    background-color: #2c3e50 !important;
                    border: 1px solid #444 !important;
                    color: #FFD3B6 !important;
                }
                .Select-value-label {
                    color: #FFD3B6 !important;
                }
                .Select-input {
                    color: #FFD3B6 !important;
                }
                .Select-arrow-zone {
                    color: #FFD3B6 !important;
                }
                .list-group-item {
                    font-size: 18px;
                    background-color: #2c3e50;
                    border: 1px solid #444;
                    margin-bottom: 0.5rem;
                }
                .card-title {
                    font-size: 1.3rem;
                    font-weight: 500;
                    margin-bottom: 1rem;
                }
                .card-body p {
                    font-size: 18px;
                    margin-bottom: 0.75rem;
                }
                .tab-content {
                    font-size: 18px;
                }
                .refresh-button {
                    margin-left: 10px;
                    background-color: #6B5B95;
                    border-color: #6B5B95;
                    color: white;
                }
                .refresh-button:hover {
                    background-color: #7D6BA0;
                    border-color: #7D6BA0;
                    color: white;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    app.layout = dbc.Container([
        dcc.Store(id='recommendations-store', data=strategic_recommendations),
        dcc.Store(id='seasonal-store', data=seasonal_stats.to_dict('records')),
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1('Airline Customer Segmentation Dashboard', className='text-center mb-3'),
                    html.H5('SmartSegment: Real-time Customer Segmentation ', className='text-center text-muted')
                ], className='dashboard-header')
            ], width=12)
        ]),
        
        # Navigation Tabs
        dbc.Tabs([
            # Overview Tab
            dbc.Tab(label='Overview', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Key Metrics', className='mb-4'),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Div('Total Customers', className='metric-title'),
                                        html.Div(f"{total_customers:,}", className='metric-value')
                                    ], className='metric-card')
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Div('Avg Ticket Price', className='metric-title'),
                                        html.Div(f"${avg_ticket_price:,.2f}", className='metric-value')
                                    ], className='metric-card')
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Div('Total Revenue', className='metric-title'),
                                        html.Div(f"${total_revenue:,.0f}", className='metric-value')
                                    ], className='metric-card')
                                ], width=3),
                                dbc.Col([
                                    html.Div([
                                        html.Div('Peak Season', className='metric-title'),
                                        html.Div(peak_season, className='metric-value')
                                    ], className='metric-card')
                                ], width=3)
                            ])
                        ], className='graph-container')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Customer Segments Distribution', className='mb-4'),
                            dcc.Graph(
                                figure=px.pie(fm, names='Cluster', title='Distribution of Customer Segments',
                                            labels={'Cluster': 'Segment'},
                                            color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H3('Frequency-Monetary Analysis', className='mb-4'),
                            dcc.Graph(
                                figure=px.scatter(fm, x='Frequency', y='Monetary', color='Cluster',
                                                title='Customer Segments by Frequency and CLV',
                                                hover_data=['Avg_Ticket_Price', 'Ancillary', 'Total_Revenue'],
                                                color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Revenue by Customer Segment', className='mb-4'),
                            dcc.Graph(
                                figure=px.bar(cluster_stats, x='Cluster', y='Total_Revenue',
                                            title='Total Revenue Contribution by Segment',
                                            labels={'Total_Revenue': 'Total Revenue ($)', 'Cluster': 'Segment'},
                                            color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=12)
                ])
            ]),
            
            # Loyalty Tier Analysis Tab
            dbc.Tab(label='Loyalty Tiers', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Loyalty Tier Distribution', className='mb-4'),
                            dcc.Graph(
                                figure=px.pie(loyalty_tier_stats, 
                                            names='Loyalty Card', 
                                            values='Loyalty Number',
                                            title='Distribution of Customers by Loyalty Tier',
                                            color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H3('Loyalty Tier Performance', className='mb-4'),
                            dcc.Graph(
                                figure=px.bar(loyalty_tier_stats,
                                            x='Loyalty Card',
                                            y=['Frequency', 'Monetary', 'Avg_Ticket_Price'],
                                            title='Key Metrics by Loyalty Tier',
                                            barmode='group',
                                            labels={'value': 'Value', 'variable': 'Metric'},
                                            color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Loyalty Tier vs Customer Segments', className='mb-4'),
                            dcc.Graph(
                                figure=px.sunburst(fm, 
                                                  path=['Loyalty Card', 'Cluster'], 
                                                  values='Total_Revenue',
                                                  title='Revenue Distribution by Loyalty Tier and Segment',
                                                  color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Loyalty Tier Insights', className='mb-4'),
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Star Tier", className="card-title"),
                                    html.P("Basic loyalty tier with the largest customer base but lowest average CLV"),
                                    html.P(f"Average Flights: {loyalty_tier_stats[loyalty_tier_stats['Loyalty Card'] == 'Star']['Frequency'].values[0]:.1f}"),
                                    html.P(f"Average CLV: ${loyalty_tier_stats[loyalty_tier_stats['Loyalty Card'] == 'Star']['Monetary'].values[0]:,.0f}")
                                ])
                            ], className="mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Aurora Tier", className="card-title"),
                                    html.P("Mid-tier loyalty with balanced frequency and value"),
                                    html.P(f"Average Flights: {loyalty_tier_stats[loyalty_tier_stats['Loyalty Card'] == 'Aurora']['Frequency'].values[0]:.1f}"),
                                    html.P(f"Average CLV: ${loyalty_tier_stats[loyalty_tier_stats['Loyalty Card'] == 'Aurora']['Monetary'].values[0]:,.0f}")
                                ])
                            ], className="mb-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Nova Tier", className="card-title"),
                                    html.P("Premium loyalty tier with highest average CLV and ticket price"),
                                    html.P(f"Average Flights: {loyalty_tier_stats[loyalty_tier_stats['Loyalty Card'] == 'Nova']['Frequency'].values[0]:.1f}"),
                                    html.P(f"Average CLV: ${loyalty_tier_stats[loyalty_tier_stats['Loyalty Card'] == 'Nova']['Monetary'].values[0]:,.0f}")
                                ])
                            ])
                        ], className='graph-container')
                    ], width=12)
                ])
            ]),
            
            # Seasonal Analysis Tab
            dbc.Tab(label='Seasonal Analysis', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Monthly Flight Distribution', className='mb-4'),
                            dcc.Graph(
                                figure=px.line(df.groupby(['Year', 'Month'])['Total Flights'].sum().reset_index(),
                                              x='Month', y='Total Flights', color='Year',
                                              title='Flight Patterns Throughout the Year',
                                              labels={'Total Flights': 'Total Number of Flights'},
                                              color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=12),
                    dbc.Col([
                        html.Div([
                            html.H3('Seasonal Revenue Analysis', className='mb-4'),
                            dcc.Graph(
                                figure=px.bar(seasonal_stats,
                                             x='Season',
                                             y=['Total Flights', 'Dollar Cost Points Redeemed'],
                                             title='Seasonal Performance Metrics',
                                             barmode='group',
                                             labels={'value': 'Amount', 'variable': 'Metric'},
                                             color_discrete_sequence=peach_skyline_colors)
                            )
                        ], className='graph-container')
                    ], width=12),
                    dbc.Col([
                        html.Div([
                            html.H4('Key Seasonal Insights', className='mb-3'),
                            dbc.Card([
                                dbc.CardBody([
                                    html.P(f'Peak Travel Months: {", ".join([calendar.month_name[m] for m in peak_months])}', className='mb-2'),
                                    html.P(f'Highest Activity Season: {peak_season}', className='mb-2'),
                                    html.P(f'Total Revenue: ${seasonal_stats["Dollar Cost Points Redeemed"].sum():,.0f}')
                                ])
                            ], className='mt-4')
                        ], className='graph-container')
                    ], width=12)
                ])
            ]),
            
            # Churn Analysis Tab
            dbc.Tab(label='Churn Analysis', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Churn Risk Analysis', className='mb-4'),
                            dcc.Graph(
                                figure=px.histogram(
                                    fm,
                                    x='Cluster',
                                    color='Churn_Risk',
                                    barmode='stack',
                                    title='Customer Churn Risk by Segment',
                                    labels={'count': 'Number of Customers', 'Cluster': 'Segment'},
                                    color_discrete_sequence=peach_skyline_colors
                                )
                            )
                        ], className='graph-container')
                    ], width=12)
                ])
            ]),
            
            # Strategic Recommendations Tab
            dbc.Tab(label='Strategic Recommendations', children=[
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3('Select Customer Segment', className='mb-4'),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='segment-dropdown',
                                        options=[
                                            {'label': f'Cluster {i} - {strategic_recommendations[f"Cluster {i}"]["title"]}',
                                            'value': f'Cluster {i}'} for i in range(4)
                                        ],
                                        value='Cluster 0',
                                        className='mb-4',
                                        style={
                                            'color': '#FFD3B6',
                                            'backgroundColor': '#2c3e50',
                                            'border': '1px solid #444'
                                        }
                                    )
                                ], width=10),
                                dbc.Col([
                                    dbc.Button("Refresh", id="refresh-button", color="primary", className="refresh-button")
                                ], width=2)
                            ])
                        ], className='graph-container')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Segment Overview", className="mb-0")),
                            dbc.CardBody([
                                html.Div(id='segment-overview')
                            ])
                        ], className='mb-3')
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Pricing Strategy", className="mb-0")),
                            dbc.CardBody([
                                html.Div(id='pricing-strategy')
                            ])
                        ], className='mb-3')
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Recommended Actions", className="mb-0")),
                            dbc.CardBody([
                                html.Div(id='recommendations-output')
                            ])
                        ], className='mb-3')
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Upsell Opportunities", className="mb-0")),
                            dbc.CardBody([
                                html.Div(id='upsell-opportunities')
                            ])
                        ])
                    ], width=6)
                ])
            ])
        ], className='mb-4')
    ], fluid=True)

    # Callbacks for interactive elements
    @app.callback(
        [Output('segment-overview', 'children'),
         Output('recommendations-output', 'children'),
         Output('pricing-strategy', 'children'),
         Output('upsell-opportunities', 'children')],
        [Input('segment-dropdown', 'value'),
         Input('refresh-button', 'n_clicks')],
        [State('recommendations-store', 'data'),
         State('seasonal-store', 'data'),
         State('segment-dropdown', 'value')]
    )
    def update_recommendations(selected_segment, n_clicks, recommendations_data, seasonal_data, current_segment):
        # Convert seasonal data back to DataFrame
        seasonal_stats = pd.DataFrame(seasonal_data)
        
        # Get peak season
        peak_season = seasonal_stats.iloc[seasonal_stats['Total Flights'].idxmax()]['Season']
        
        # Regenerate recommendations when refresh button is clicked
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'refresh-button.n_clicks':
            strategic_recommendations = generate_dynamic_recommendations(cluster_stats, peak_season, peak_months)
            selected_segment = current_segment  # Keep the same segment selected
        else:
            strategic_recommendations = recommendations_data
        
        segment = strategic_recommendations[selected_segment]
        cluster_num = int(selected_segment.split()[-1])
        stats = cluster_stats[cluster_stats['Cluster'] == cluster_num].iloc[0]
        
        peak_months_str = ', '.join([calendar.month_name[m] for m in peak_months])
        
        overview = html.Div([
            html.P(f"Segment: {segment['title']}", className='mb-2'),
            html.P(f"Revenue Contribution: {segment['revenue_contribution']}", className='mb-2'),
            html.P(f"Average CLV: ${stats['Monetary']:,.0f}", className='mb-2'),
            html.P(f"Average Flights: {stats['Frequency']:.1f} per year", className='mb-2'),
            html.P(f"Average Ticket Price: ${stats['Avg_Ticket_Price']:.2f}", className='mb-2'),
            html.P(f"Peak Travel Season: {peak_season}", className='mb-2'),
            html.P(f"Peak Travel Months: {peak_months_str}")
        ])
        
        recommendations = html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem(rec, className='mb-2') for rec in segment['recommendations']
            ])
        ])
        
        pricing = html.Div([
            html.P(f"Price Sensitivity: {segment['price_sensitivity']}", className='mb-2'),
            html.P(f"Average Ticket Price: ${stats['Avg_Ticket_Price']:.2f}", className='mb-2'),
            html.P("Recommended Strategy:", className='mb-2'),
            html.Ul([
                html.Li(f"Seasonal pricing adjustments for {peak_season} (peak season)", className='mb-2'),
                html.Li("5-10% discounts for price-sensitive segments", className='mb-2') if "High" in segment['price_sensitivity'] else None,
                html.Li("Value-added services for less sensitive segments", className='mb-2') if "Low" in segment['price_sensitivity'] else None,
                html.Li("Dynamic pricing based on booking patterns and seasonal demand")
            ])
        ])
        
        upsell = html.Div([
            html.P("Top Upsell Opportunities:", className='mb-2'),
            html.Ul([html.Li(item, className='mb-2') for item in segment['upsell_opportunities']])
        ])
        
        return overview, recommendations, pricing, upsell
    
    return app

if __name__ == "__main__":
    # Preprocess data with available columns
    fm, df, seasonal_stats, peak_months, loyalty_tier_stats = preprocess_data()
    
    # Analyze business cases with enhanced metrics
    fm = analyze_business_cases(fm, df)
    
    # Create enhanced dashboard
    app = create_dashboard(fm, df, seasonal_stats, peak_months, loyalty_tier_stats)
    app.run(debug=True)