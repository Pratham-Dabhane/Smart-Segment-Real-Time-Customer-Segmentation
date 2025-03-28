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

# Set display options for better visualization
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Set random seed for reproducibility
np.random.seed(42)

# Load the datasets
flight_activity = pd.read_csv('dataset/Customer Flight Activity.csv')
loyalty_history = pd.read_csv('dataset/Customer Loyalty History.csv')

# Enhanced Data Preprocessing with available columns
def preprocess_data():
    # Merge the datasets
    df = pd.merge(flight_activity, loyalty_history, on='Loyalty Number', how='inner')
    
    # Calculate average ticket price (approximation)
    df['Avg_Ticket_Price'] = df['CLV'] / df['Total Flights']
    
    # Calculate ancillary revenue (approximation)
    df['Ancillary'] = df['Points Redeemed'] * 0.02  # Assuming $0.02 value per point
    
    # Calculate FM metrics (Frequency and Monetary)
    frequency = df.groupby('Loyalty Number')['Total Flights'].sum().reset_index()
    monetary = df.groupby('Loyalty Number')['CLV'].first().reset_index()
    ancillary = df.groupby('Loyalty Number')['Ancillary'].mean().reset_index()
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
    seasonal_stats = df.groupby('Season').agg({
        'Total Flights': 'sum',
        'CLV': 'mean',
        'Points Accumulated': 'sum'
    }).reset_index()
    
    # Combine metrics
    fm = pd.merge(frequency, monetary, on='Loyalty Number')
    fm = pd.merge(fm, ancillary, on='Loyalty Number')
    fm = pd.merge(fm, avg_price, on='Loyalty Number')
    fm.columns = ['Loyalty Number', 'Frequency', 'Monetary', 'Ancillary', 'Avg_Ticket_Price']
    
    # Add loyalty card type and other relevant features
    loyalty_card = df.groupby('Loyalty Number')['Loyalty Card'].first().reset_index()
    fm = pd.merge(fm, loyalty_card, on='Loyalty Number')
    
    return fm, df, seasonal_stats, peak_months

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
    fm['Total_Revenue'] = fm['Avg_Ticket_Price'] * fm['Frequency']
    fm['Revenue_with_Ancillary'] = fm['Total_Revenue'] + fm['Ancillary']
    
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

# Enhanced Dashboard with new features
def create_dashboard(fm, df, seasonal_stats, peak_months):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Calculate cluster statistics for new metrics
    cluster_stats = fm.groupby('Cluster').agg({
        'Monetary': 'mean',
        'Frequency': 'mean',
        'Total_Revenue': 'sum',
        'Revenue_with_Ancillary': 'sum',
        'Avg_Ticket_Price': 'mean',
        'Ancillary': 'mean',
        'Loyalty Number': 'count'
    }).reset_index()
    cluster_stats['Revenue_Contribution'] = (cluster_stats['Total_Revenue'] / cluster_stats['Total_Revenue'].sum()) * 100
    cluster_stats['Segment_Size'] = (cluster_stats['Loyalty Number'] / cluster_stats['Loyalty Number'].sum()) * 100
    
    # Calculate loyalty tier statistics
    loyalty_stats = fm.groupby('Loyalty Card').agg({
        'Monetary': 'mean',
        'Frequency': 'mean',
        'Total_Revenue': 'sum',
        'Revenue_with_Ancillary': 'sum',
        'Loyalty Number': 'count'
    }).reset_index()
    loyalty_stats['ROI'] = (loyalty_stats['Total_Revenue'] / loyalty_stats['Total_Revenue'].sum()) * 100
    
    # Strategic recommendations with new metrics
    strategic_recommendations = {
        'Cluster 0': {
            'title': 'High Frequency, Moderate Value Customers',
            'revenue_contribution': f"{cluster_stats.loc[0, 'Revenue_Contribution']:.1f}%",
            'recommendations': [
                'Implement tiered pricing based on booking frequency',
                'Offer bulk booking discounts (5-10% for 3+ flights)',
                'Introduce family travel packages using companion flight data',
                'Corporate partnership programs targeting frequent flyers'
            ],
            'price_sensitivity': 'Moderate (responds to 5-10% discounts)',
            'upsell_opportunities': ['Family packages', 'Group discounts', 'Seasonal passes']
        },
        'Cluster 1': {
            'title': 'Low Frequency, Moderate Value Customers',
            'revenue_contribution': f"{cluster_stats.loc[1, 'Revenue_Contribution']:.1f}%",
            'recommendations': [
                'Create seasonal travel promotions (20-30% off)',
                'Weekend getaway packages with hotel partnerships',
                'Flexible booking options with 10% premium',
                'Loyalty program sign-up bonuses (500 bonus miles)'
            ],
            'price_sensitivity': 'High (needs 20-30% discounts)',
            'upsell_opportunities': ['Hotel bundles', 'Car rentals', 'Travel insurance']
        },
        'Cluster 2': {
            'title': 'High Value Customers',
            'revenue_contribution': f"{cluster_stats.loc[2, 'Revenue_Contribution']:.1f}%",
            'recommendations': [
                'Premium seat upgrades based on CLV',
                'Personalized travel concierge services',
                'VIP event partnerships (sports, concerts)',
                'Dedicated account managers for top customers'
            ],
            'price_sensitivity': 'Low (willing to pay 15-20% premium)',
            'upsell_opportunities': ['First class upgrades', 'Lounge memberships', 'Priority services']
        },
        'Cluster 3': {
            'title': 'Medium-High Frequency, High Value Customers',
            'revenue_contribution': f"{cluster_stats.loc[3, 'Revenue_Contribution']:.1f}%",
            'recommendations': [
                'Dynamic pricing based on travel patterns',
                'Premium loyalty tiers (Gold/Platinum)',
                'Business travel packages using companion data',
                'Exclusive destination partnerships'
            ],
            'price_sensitivity': 'Moderate-Low (responds to value-added services)',
            'upsell_opportunities': ['Business class upgrades', 'Airport transfers', 'Premium meals']
        }
    }
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1('Airline Customer Segmentation Dashboard', className='text-center my-4'), width=12)
        ]),
        
        # Navigation Tabs
        dbc.Tabs([
            # Seasonal Analysis Tab
            dbc.Tab(label='Seasonal Analysis', children=[
                dbc.Row([
                    dbc.Col([
                        html.H3('Monthly Flight Distribution'),
                        dcc.Graph(
                            figure=px.line(df.groupby(['Year', 'Month'])['Total Flights'].sum().reset_index(),
                                          x='Month', y='Total Flights', color='Year',
                                          title='Flight Patterns Throughout the Year',
                                          labels={'Total Flights': 'Total Number of Flights'})
                        )
                    ], width=12),
                    dbc.Col([
                        html.H3('Seasonal Revenue Analysis'),
                        dcc.Graph(
                            figure=px.bar(seasonal_stats,
                                         x='Season',
                                         y=['Total Flights', 'CLV', 'Points Accumulated'],
                                         title='Seasonal Performance Metrics',
                                         barmode='group')
                        )
                    ], width=12),
                    dbc.Col([
                        html.Div([
                            html.H4('Key Seasonal Insights'),
                            html.P(f'Peak Travel Months: {", ".join([calendar.month_name[m] for m in peak_months])}'),
                            html.P(f'Highest Activity Season: {seasonal_stats.nlargest(1, "Total Flights")["Season"].iloc[0]}'),
                            html.P(f'Average Revenue per Season: ${seasonal_stats["CLV"].mean():,.2f}')
                        ], className='mt-4')
                    ], width=12)
                ])
            ]),

            # Overview Tab
            dbc.Tab(label='Overview', children=[
                dbc.Row([
                    dbc.Col([
                        html.H3('Customer Segments Distribution'),
                        dcc.Graph(
                            figure=px.pie(fm, names='Cluster', title='Distribution of Customer Segments',
                                        labels={'Cluster': 'Segment'})
                        )
                    ], width=6),
                    dbc.Col([
                        html.H3('Frequency-Monetary Analysis'),
                        dcc.Graph(
                            figure=px.scatter(fm, x='Frequency', y='Monetary', color='Cluster',
                                            title='Customer Segments by Frequency and CLV',
                                            hover_data=['Avg_Ticket_Price', 'Ancillary', 'Total_Revenue'])
                        )
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3('Revenue Analysis by Segment'),
                        dcc.Graph(
                            figure=px.bar(cluster_stats, x='Cluster', y='Total_Revenue',
                                        title='Total Revenue by Customer Segment',
                                        labels={'Total_Revenue': 'Total Revenue ($)', 'Cluster': 'Segment'},
                                        text='Revenue_Contribution',
                                        hover_data=['Revenue_Contribution', 'Segment_Size'])
                        )
                    ], width=6),
                    dbc.Col([
                        html.H3('Ancillary Revenue Potential'),
                        dcc.Graph(
                            figure=px.bar(cluster_stats, x='Cluster', y='Ancillary',
                                        title='Average Ancillary Revenue per Customer',
                                        labels={'Ancillary': 'Ancillary Revenue ($)', 'Cluster': 'Segment'})
                        )
                    ], width=6)
                ])
            ]),
            
            # Revenue & Pricing Tab
            dbc.Tab(label='Revenue & Pricing', children=[
                dbc.Row([
                    dbc.Col([
                        html.H3('Revenue Breakdown by Segment'),
                        dcc.Graph(
                            figure=px.sunburst(
                                fm,
                                path=['Cluster', 'Loyalty Card'],
                                values='Total_Revenue',
                                title='Revenue Distribution by Segment and Loyalty Tier'
                            )
                        )
                    ], width=6),
                    dbc.Col([
                        html.H3('Pricing Sensitivity Analysis'),
                        dcc.Graph(
                            figure=px.scatter(
                                fm, 
                                x='Avg_Ticket_Price', 
                                y='Frequency', 
                                color='Cluster',
                                title='Ticket Price vs. Flight Frequency',
                                # trendline="lowess",  # Removed as it requires statsmodels
                                labels={
                                    'Avg_Ticket_Price': 'Average Ticket Price ($)',
                                    'Frequency': 'Flights per Year'
                                }
                            )
                        )
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3('Revenue Contribution by Segment'),
                        dash_table.DataTable(
                            id='revenue-table',
                            columns=[
                                {"name": "Segment", "id": "Cluster"},
                                {"name": "Revenue", "id": "Total_Revenue", "type": "numeric", "format": {"specifier": "$.2f"}},
                                {"name": "Contribution", "id": "Revenue_Contribution", "type": "numeric", "format": {"specifier": ".1f"}},
                                {"name": "Avg. Ticket Price", "id": "Avg_Ticket_Price", "type": "numeric", "format": {"specifier": "$.2f"}},
                                {"name": "Customers", "id": "Loyalty Number"}
                            ],
                            data=cluster_stats.to_dict('records'),
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            }
                        )
                    ], width=12)
                ])
            ]),
            
            # Loyalty & Upselling Tab
            dbc.Tab(label='Loyalty & Upselling', children=[
                dbc.Row([
                    dbc.Col([
                        html.H3('Loyalty Program ROI Analysis'),
                        dcc.Graph(
                            figure=px.bar(
                                loyalty_stats,
                                x='Loyalty Card',
                                y='ROI',
                                color='Loyalty Card',
                                title='Return on Investment by Loyalty Tier',
                                labels={'ROI': 'Revenue Contribution (%)', 'Loyalty Card': 'Loyalty Tier'}
                            )
                        )
                    ], width=6),
                    dbc.Col([
                        html.H3('Ancillary Revenue Opportunities'),
                        dcc.Graph(
                            figure=px.box(
                                fm,
                                x='Cluster',
                                y='Ancillary',
                                color='Cluster',
                                title='Ancillary Revenue Potential by Segment',
                                labels={'Ancillary': 'Ancillary Revenue ($)', 'Cluster': 'Segment'}
                            )
                        )
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3('Churn Risk Analysis'),
                        dcc.Graph(
                            figure=px.histogram(
                                fm,
                                x='Cluster',
                                color='Churn_Risk',
                                barmode='stack',
                                title='Customer Churn Risk by Segment',
                                labels={'count': 'Number of Customers', 'Cluster': 'Segment'}
                            )
                        )
                    ], width=12)
                ])
            ]),
            
            # Strategic Recommendations Tab
            dbc.Tab(label='Strategic Recommendations', children=[
                dbc.Row([
                    dbc.Col([
                        html.H3('Select Customer Segment'),
                        dcc.Dropdown(
                            id='segment-dropdown',
                            options=[
                                {'label': f'Cluster {i} - {strategic_recommendations[f"Cluster {i}"]["title"]}',
                                 'value': f'Cluster {i}'} for i in range(4)
                            ],
                            value='Cluster 0'
                        )
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
            ]),
            
            # Action Planning Tab
            dbc.Tab(label='Action Planning', children=[
                dbc.Row([
                    dbc.Col([
                        html.H3('Revenue Growth Initiatives'),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4('Short-term Actions (0-3 months)'),
                                html.Ul([
                                    html.Li(html.Strong('Targeted promotions:'), 
                                    " Implement cluster-specific email campaigns with personalized offers"),
                                    html.Li(html.Strong('Dynamic pricing:'), 
                                    " Adjust prices based on segment price sensitivity"),
                                    html.Li(html.Strong('Ancillary bundles:'), 
                                    " Create package deals for high-potential segments")
                                ])
                            ])
                        ], className='mb-3'),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4('Medium-term Actions (3-6 months)'),
                                html.Ul([
                                    html.Li(html.Strong('Loyalty program redesign:'), 
                                    " Align rewards with segment value and behavior"),
                                    html.Li(html.Strong('Route optimization:'), 
                                    " Adjust flight frequencies based on segment demand"),
                                    html.Li(html.Strong('Partnership development:'), 
                                    " Create hotel and car rental packages for target segments")
                                ])
                            ])
                        ], className='mb-3'),
                        dbc.Card([
                            dbc.CardBody([
                                html.H4('Long-term Actions (6-12 months)'),
                                html.Ul([
                                    html.Li(html.Strong('Premium services:'), 
                                    " Develop concierge services for high-value segments"),
                                    html.Li(html.Strong('Predictive analytics:'), 
                                    " Implement AI-driven pricing and churn prediction"),
                                    html.Li(html.Strong('Customer journey redesign:'), 
                                    " Personalize all touchpoints based on segment characteristics")
                                ])
                            ])
                        ])
                    ], width=12)
                ])
            ])
        ])
    ], fluid=True)
    
    # Callbacks for interactive elements
    @app.callback(
        [Output('segment-overview', 'children'),
         Output('recommendations-output', 'children'),
         Output('pricing-strategy', 'children'),
         Output('upsell-opportunities', 'children')],
        Input('segment-dropdown', 'value')
    )
    def update_recommendations(selected_segment):
        segment = strategic_recommendations[selected_segment]
        cluster_num = int(selected_segment.split()[-1])
        stats = cluster_stats[cluster_stats['Cluster'] == cluster_num].iloc[0]
        
        # Get seasonal insights
        peak_season = seasonal_stats.nlargest(1, 'Total Flights')['Season'].iloc[0]
        peak_months_str = ', '.join([calendar.month_name[m] for m in peak_months])
        
        overview = html.Div([
            html.P(f"Segment: {segment['title']}"),
            html.P(f"Revenue Contribution: {segment['revenue_contribution']}"),
            html.P(f"Average CLV: ${stats['Monetary']:,.0f}"),
            html.P(f"Average Flights: {stats['Frequency']:.1f} per year"),
            html.P(f"Ancillary Revenue: ${stats['Ancillary']:.2f} per customer"),
            html.P(f"Peak Travel Season: {peak_season}"),
            html.P(f"Peak Travel Months: {peak_months_str}")
        ])
        
        # Dynamic recommendations based on seasonal patterns
        dynamic_recommendations = segment['recommendations'].copy()
        if peak_season == 'Summer':
            dynamic_recommendations.append(f"Special summer vacation packages targeting {segment['title']}")
        elif peak_season == 'Winter':
            dynamic_recommendations.append(f"Winter holiday promotions for {segment['title']}")
        
        recommendations = html.Div([
            dbc.ListGroup([
                dbc.ListGroupItem(rec) for rec in dynamic_recommendations
            ])
        ])
        
        # Dynamic pricing strategy based on seasonal demand
        pricing = html.Div([
            html.P(f"Price Sensitivity: {segment['price_sensitivity']}"),
            html.P(f"Average Ticket Price: ${stats['Avg_Ticket_Price']:.2f}"),
            html.P("Recommended Strategy:"),
            html.Ul([
                html.Li(f"Seasonal pricing adjustments for {peak_season} (peak season)"),
                html.Li("5-10% discounts for price-sensitive segments") if "High" in segment['price_sensitivity'] else None,
                html.Li("Value-added services for less sensitive segments") if "Low" in segment['price_sensitivity'] else None,
                html.Li("Dynamic pricing based on booking patterns and seasonal demand")
            ])
        ])
        
        # Dynamic upsell opportunities based on season
        seasonal_upsell = segment['upsell_opportunities'].copy()
        if peak_season in ['Summer', 'Spring']:
            seasonal_upsell.append('Vacation packages')
        elif peak_season in ['Fall', 'Winter']:
            seasonal_upsell.append('Holiday travel bundles')
        
        upsell = html.Div([
            html.P("Top Upsell Opportunities:"),
            html.Ul([html.Li(item) for item in seasonal_upsell])
        ])
        
        return overview, recommendations, pricing, upsell
    
    app.run(debug=True)

def main():
    # Preprocess data with available columns
    fm, df, seasonal_stats, peak_months = preprocess_data()
    
    # Analyze business cases with enhanced metrics
    fm = analyze_business_cases(fm, df)
    
    # Create enhanced dashboard
    create_dashboard(fm, df)

if __name__ == "__main__":
    fm, df, seasonal_stats, peak_months = preprocess_data()
    fm = analyze_business_cases(fm, df)
    create_dashboard(fm, df, seasonal_stats, peak_months)