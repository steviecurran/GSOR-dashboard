#!/usr/bin/python3  # REMOVE FOR pythonanywhere
dir1 = "App2"; dir2 = "~/courses/BI_Udemy/python/maps/" #HOME
#dir1 = "/home/steviecurran/mysite/App2/"; dir2 = "/home/steviecurran/maps/" # pythonanywhere

import numpy as np
import pandas as pd
import os 
import sys
import calendar
import datetime as dt
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0]) 
pd.set_option('display.max_columns', None)

import dash
from dash import Dash, html, dcc, Input, Output, ctx, State,dash_table
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots # FOR SECONDARY AXIS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None  # default='warn'
from forex_python.converter import CurrencyRates
c = CurrencyRates()

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions=True  


###########################  CURRENCIES ##################################
cur =  pd.read_csv(dir1+'/codes_crop.csv'); #print(cur)
cur  = cur.sort_values('AlphabeticCode')
currs = cur['AlphabeticCode'].unique(); #print(currs)
cur_dict = dict(zip(cur.AlphabeticCode,cur.Currency)); #print(cur_dict)
###################################################
df = pd.read_csv(dir1+'/orders.csv');#print(df)
no_orders = df['Quantity'].count();

# Returned Items Value  THIS IS TRICKY, AS NEED Order ID FROM
df2 = pd.read_csv(dir1+'/return.csv'); #print(df2)
df3 = pd.merge(df2, df, on = ['Order ID'], how = 'outer');#print("df3\n", df3)
df3['Returned'] = df3['Returned'].fillna('No'); dfsandr = df3.copy()

total_sales = len(df3)
returns = df3[df3['Returned']== "Yes"]; #print(returns)
riv =  round(returns['Sales'].sum(),2); #print(riv)
no_ret = len(returns)
#######################  THREE PLOTS ###############
ship = df.groupby(['Ship Mode']).count()['Order ID'].to_frame(name = 'Count').reset_index()
seg = df.groupby(['Segment']).sum()['Sales'].to_frame(name = 'Sum').reset_index(); #print(seg)
dft = df.groupby(['Order Priority']).count()['Order ID'].to_frame(name = 'Count').reset_index();
#print(dft)
######################### TABLE ###############
def cal(datetype):
        df[datetype] = pd.to_datetime(df[datetype],infer_datetime_format=True);
        df['Day'] =  df[datetype].dt.dayofweek  # Monday is 0 and Sunday is 6 
        df['Month'] = df[datetype].dt.month
        df['Year'] = df[datetype].dt.year
cal('Order Date')

def table_year(p):
        dfp = df[df['Order Priority'] == p]
        df1= dfp.groupby(['Year']).count()['Order ID'].to_frame(name = 'No orders').reset_index()
        df2= dfp.groupby(['Year']).sum()['Sales'].to_frame(name = 'Sales Amount').reset_index();#print(df2)
        df3 = df1.merge(df2,on = 'Year'); #print(df3)
        return df3

df1 = table_year('Medium');df1 = df1.rename({'No orders': 'No orders_1', 'Sales Amount': 'Sales Amount_1'}, axis=1); 
df2 = table_year('Low'); df2 = df2.rename({'No orders': 'No orders_2', 'Sales Amount': 'Sales Amount_2'}, axis=1); 
df12 = df2.merge(df1,on = 'Year'); 
df3 = table_year('High'); df3 = df3.rename({'No orders': 'No orders_3', 'Sales Amount': 'Sales Amount_3'}, axis=1);
df123 = df3.merge(df12,on = 'Year'); df4 = table_year('Critical')
df1234 = df4.merge(df123,on = 'Year'); 
df1234['Sales Amount'] = round(df1234['Sales Amount'],2)
df1234['Sales Amount_1'] = round(df1234['Sales Amount_1'],2)
df1234['Sales Amount_2'] = round(df1234['Sales Amount_2'],2)
df1234['Sales Amount_3'] = round(df1234['Sales Amount_3'],2)
df1234['Orders total'] = df1234['No orders'] + df1234['No orders_1'] + df1234['No orders_2'] + df1234['No orders_3']
df1234['Sales total'] = round(df1234['Sales Amount'] + df1234['Sales Amount_1'] + df1234['Sales Amount_2'] + df1234['Sales Amount_3'],2);#print(df1234)


def table_month(p,year):
        dfy = df[df['Year'] == year]
        dfp = dfy[dfy['Order Priority'] == p]
        df1= dfp.groupby(['Month']).count()['Order ID'].to_frame(name = 'No orders').reset_index()
        df2= dfp.groupby(['Month']).sum()['Sales'].to_frame(name = 'Sales Amount').reset_index();#print(df2)
        df3 = df1.merge(df2,on = 'Month');
        df3['Month Name'] = pd.to_datetime(df3['Month'], format='%m').dt.month_name(); #print(df3)
        return df3

def merge_month(year):
        df1 = table_month('Medium',year);df1 = df1.rename({'No orders': 'No orders_1', 'Sales Amount': 'Sales Amount_1'}, axis=1); 
        df2 = table_month('Low',year); df2 = df2.rename({'No orders': 'No orders_2', 'Sales Amount': 'Sales Amount_2'}, axis=1); 
        df12 = df2.merge(df1,on = 'Month Name'); 
        df3 = table_month('High',year); df3 = df3.rename({'No orders': 'No orders_3', 'Sales Amount': 'Sales Amount_3'}, axis=1);
        df123 = df3.merge(df12,on = 'Month Name');
        df4 = table_month('Critical',year);
        df1234 = df4.merge(df123,on = 'Month Name');
        df1234['Sales Amount'] = round(df1234['Sales Amount'],2)
        df1234['Sales Amount_1'] = round(df1234['Sales Amount_1'],2)
        df1234['Sales Amount_2'] = round(df1234['Sales Amount_2'],2)
        df1234['Sales Amount_3'] = round(df1234['Sales Amount_3'],2); 
        df1234['Orders total'] = df1234['No orders'] + df1234['No orders_1'] + df1234['No orders_2'] + df1234['No orders_3']
        df1234['Sales total'] = round(df1234['Sales Amount'] + df1234['Sales Amount_1'] + df1234['Sales Amount_2'] + df1234['Sales Amount_3'],2);
        return df1234

########################## PAGE 2 ####################################
#Returned Orders by Segment
dfa = pd.read_csv(dir1+'/orders.csv')
dfb = pd.read_csv(dir1+'/return.csv'); # REREAD, JUST IN CASE, RENAMED AS WAS CLASHING WITH merge_month(year)
dfc = dfb.merge(dfa,on = 'Order ID'); 
dfd = dfc[dfc['Returned'] == "Yes"]; #print(dfd)

seg_ret = dfd.groupby(['Segment']).count()['Sales'].to_frame(name = 'Number').reset_index();#print(seg_ret)
mark_ret = dfd.groupby(['Market']).count()['Sales'].to_frame(name = 'Number').reset_index();#print(mark_ret)
mark_sal = dfd.groupby(['Market']).sum()['Sales'].to_frame(name = 'Total').reset_index();#print(mark_sal)
mark_stack = mark_sal.merge(mark_ret,on = 'Market');
mark_stack = mark_stack.sort_values('Number',ascending=False) # TO OVERLAY LINE ON ASCENDING BARS

pri_ret = dfd.groupby(['Order Priority']).count()['Sales'].to_frame(name = 'Number').reset_index();
pri_ret = pri_ret.sort_values('Number',ascending=True)
#print(pri_ret)

pd.options.mode.chained_assignment = None  # default='warn'
### Map for % Returned Items by Country  ###
dfcodes = pd.read_csv(dir2+'/2014_world_gdp_with_codes.csv');
dfcodes.rename(columns={'COUNTRY':'Country'}, inplace = True);
dfr = dfsandr.merge(dfcodes,on = 'Country');
dfr = dfr.sort_values('CODE')
couns = list(dfr['CODE'].unique());
df_coun= pd.DataFrame();
yes = dfr[dfr['Returned'] == "Yes"]; 
no = dfr[dfr['Returned'] == "No"]
for (i,coun) in enumerate(couns):
        ret = yes[yes['CODE'] == coun]; yes_ret = len(ret)
        nop = no[no['CODE'] == coun]; nope_ret = len(nop)
        ret['yes_ret'] = yes_ret; ret['nope_ret'] = nope_ret;
        top = nop.iloc[0]; 
        top['perc'] = 100*float(yes_ret)/(yes_ret+nope_ret);     
        df_coun = df_coun.append(top)    

# Table visual with Customer Name, # Returned Orders and Value of Returned Products.
df_cust= pd.DataFrame();
tmp1 = yes.sort_values('Customer Name'); 
customers = tmp1['Customer ID']. unique(); 

for (i,cust) in enumerate(customers):
        tmp2 = tmp1[tmp1['Customer ID']==cust]; 
        tmp2['No returns'] = tmp2['Quantity'].sum()
        tmp2['Value'] = tmp2['Sales'].sum(); 
        top = tmp2.iloc[0]; 
        df_cust = df_cust.append(top)

df_cust = df_cust[['Customer Name','Customer ID','No returns','Value']]
df_cust= df_cust.fillna('0')
df_cust['Value'] = round(df_cust['Value'],2)
df_cust_orig = df_cust.copy()

def rates(check):
        try:
                op= c.get_rate('USD',check)  # e.g. AWG not available for Date latest
                rate = "%1.3f" %(c.get_rate('USD', check));
                inv_rate = "%1.3f" %(c.get_rate(check,'USD'))
                curr_text = '1 US Dollar is %s %s  - 1 %s = %s USD' %(rate,cur_dict[check],check,inv_rate)
                curr = check
        
        except Exception as re:  
                #print(re)
                op=None
                rate = 1.00; inv_rate = 1.00
                curr_text = '   Latest %s [%s] not available' %(check,cur_dict[check]) 
                curr = "USD" 
        return rate,curr,curr_text

################ CURRENCY SELECTION HERE AS TOO MUCH TO-ING ANF FRO-ING ##########
SIDEBAR_STYLE = {"position": "fixed","top": 100,"left": 0,"bottom": 0,"width": "7rem","padding": "2rem 1rem"}
CONTENT_STYLE = {"margin-left": "8rem", "margin-right": "10rem", "padding": "1rem 1rem"}

sidebar = html.Div([
        "Currency",
        dcc.RadioItems(
                currs,
                'USD', # default
                id='curr-check',
                style={"display":"block"}  # NO WAY TO SQUARIFY THIS
    ),
], style=dict(SIDEBAR_STYLE,overflow= "scroll"),
                   ) # NO COMMA

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"),sidebar,content])
@app.callback(Output("page-content", "children"),[Input("url", "pathname")])


def render_page_content(pathname):
        if pathname == "/":
                return html.Div([
                        html.Br(),
                        dbc.Nav([
                                html.P("Global Superstore Orders & Returns",
                                       style={'background-color':'#1D3557','font-family':'Segoe UI',"color":'white',
                                              'padding':'5px','fontSize': 24,'font-weight':'bold',
                                              'margin-left': '300px','textAlign':'center','width':'50%'}
                                       ),

                                dbc.Button("Returns", href="/returns", active=True, 
                                           style={"background-color": "#d64550","color": "white",
                                                  'border-color': 'black', 'border-width':'3px', 
                                                  'fontSize': 16,'width':'10%','margin-left': '50px',
                                                  'padding':'5px','height':'40px','width':'80px'}
                                           ),
                        ],
                                vertical=False,
                                pills=False, # True hides inactive
                                ),

                        dbc.Row([
                                dbc.Col([
                                        html.P([(f"{no_orders:,}"), html.Br(), "Number of Orders"]),
                                ]),
                                dbc.Col([
                                        html.Div(id='sales_amount'),"Sales Amount"
                                ]), 
                                dbc.Col([
                                           html.Div(id='returns_value'),"Returns Value"
                                ]),
                                dbc.Col([
                                        html.P(["%1.2f%%" %(100*float(no_ret)/total_sales), html.Br(),
                                                "Returned Items"])
                                ]),
                        ]),

                        dbc.Row([html.P(id='text_rate', style={'margin-left': '260px', 'margin-right': '0px', 'font-weight':'bold'},)]
                                ),

                        dbc.Row([
                                    dbc.Col([
                                            html.Div([
                                                    "Number of Orders by Ship Mode"
                                            ] ,style={'background-color':'#d64550','font-family':'Segoe UI','color': 'white',
                                                      'fontSize': 16, 'textAlign':'center'}
                                                     ),
                                            dcc.Graph(id='donut1'),
                                    ],style=dict(width='35%')
                                            ),

                                    dbc.Col([
                                            html.Div([
                                                    "Sales Amount by Segment"
                                            ],style={'background-color':'#d64550','font-family':'Segoe UI','color': 'white',
                                                     'fontSize': 16, 'textAlign':'center'}
                                                     ),
                                            dcc.Graph(id='histo1', style = {'autosize':True,'height':400, 'width':420,
                                                        "margin-right": "0px"}),
                                    ],style=dict(width='35%')
                                            ),

                                    dbc.Col([
                                            html.Div([
                                                    "#Orders by Order priority"
                                            ],style={'background-color':'#d64550','font-family':'Segoe UI','color': 'white',
                                                     'fontSize': 16, 'textAlign':'center'}
                                                     ),
                                            dcc.Graph(id='tree1'),
                                    ],style=dict(width='30%')
                                            ),
                                dash_table.DataTable(
                                        id="table",
                                        columns=[
                                {"name": ["Order Priority","Year"],"id": "year"},
                                {"name": ["Critical", "Number of Orders"],"id": "c1"},
                                {"name": ["Critical", "Sales Amount"],"id": "c2"},
                                {"name": ["High", "Number of Orders"],"id": "h1"},
                                {"name": ["High", "Sales Amount"],"id": "h2"},
                                {"name": ["Low", "Number of Orders"],"id": "l1"},
                                {"name": ["Low", "Sales Amount"],"id": "l2"},
                                {"name": ["Medium", "Number of Orders"],"id": "m1"},
                                {"name": ["Medium", "Sales Amount"],"id": "m2"},
                                {"name": ["Total", "Number of Orders"],"id": "t1"},
                                {"name": ["Total", "Sales Amount"],"id": "t2"},
                            ],
                            data =[
                                {
                                "year": df1234['Year'].iloc[i],
                                "c1": df1234['No orders'].iloc[i],
                                "c2": df1234['Sales Amount'].iloc[i],
                                "h1": df1234['No orders_3'].iloc[i],
                                "h2": df1234['Sales Amount_3'].iloc[i],
                                "l1": df1234['No orders_2'].iloc[i],
                                "l2": df1234['Sales Amount_2'].iloc[i],
                                "m1": df1234['No orders_1'].iloc[i],
                                "m2": df1234['Sales Amount_1'].iloc[i],
                                "t1": df1234['Orders total'].iloc[i],
                                "t2": df1234['Sales total'].iloc[i]
                                 }
                                for i in range(0,len(df1234)) # CAN ONLY HANDLE A SINGLE NUMBER AT A TIME
                            ],
                            style_data_conditional=[{'if': {'column_id': 'year'},'background-color': 'dimgrey','color': 'white'}],
                            merge_duplicate_headers=True, 
                            is_focused=True,
                            style_as_list_view=True,
                            style_header={'backgroundColor':'dimgrey','padding':'2px','color':'white'},
                            style_cell={'backgroundColor':'white', 'padding': '2px','textAlign': 'left',
                                        'minWidth': 95, 'maxWidth': 95, 'width': 95,'font_family': 'arial',
                                        'font_size': '16px','whiteSpace':'normal','height':'auto'},
                            style_table={'width': '100%', 'overflowX':'scroll'},
                        ),
                        html.Div(id="cell-output"),

                            ]),
                                dcc.Graph(id='table-month'),
                    ])

        elif pathname == "/returns":
            return html.Div([
                html.Br(),
                    dbc.Nav([
                            html.P("Global Superstore Orders & Returns",
                                   style={'background-color':'#1D3557','font-family':'Segoe UI','color':
                                          'white','padding':'5px','fontSize': 24,'font-weight':
                                          'bold', 'margin-left': '300px','textAlign':'center','width':'50%'}
                                   ),

                            dbc.Button("Sales", href="/", active=True, 
                                        style={"background-color": "green","color": "white",
                                               'border-color': 'black', 'border-width':'3px', 
                                               'fontSize': 16,'width':'10%','margin-left': '50px',
                                               'padding':'5px','height':'40px','width':'80px'}
                                        ),
                        ],
                            vertical=False,
                            pills=False, # True hides inactive
                            ),
                    dbc.Row([
                            dbc.Col([
                                    html.P([(f"{no_orders:,}"), html.Br(), 
                                            "Number of Orders"]),
                            ]),
                            dbc.Col([
                                    html.Div(id='sales_amount2'),"Sales Amount"
                            ]),
                            dbc.Col([
                                     html.Div(id='returns_value2'),"Returns Value"
                            ]),
                            dbc.Col([
                                    html.P(["%1.2f" %(100*float(no_ret)/total_sales), html.Br(),
                                                    "% Returned Items"])
                            ]),
                            ]),
                    dbc.Row([html.P(id='text_rate2', style={'margin-left': '260px', 'margin-right': '0px', 'font-weight':'bold'},)]),
                    dbc.Row([
                            dbc.Col([
                                    html.Div([
                                            "#Returned Orders by Segment"
                                    ] ,style={'background-color':'#1D3537','font-family':'Segoe UI','color': 'white',
                                              'fontSize': 16, 'textAlign':'center'}
                                             ),

                        
                                    dcc.Graph(id='donut2'),
                            ],style=dict(width='35%')
                                    ),

                            dbc.Col([
                                    html.Div([
                                            "Returned Orders by Market"
                                    ],style={'background-color':'#1D3537','font-family':'Segoe UI','color': 'white',
                                             'fontSize': 16, 'textAlign':'center'}
                                             ),
                                    dcc.Graph(id='stack1', style = {'autosize':True,'height':400, 'width':400,
                                                                    "margin-right": "0px"}),
                            ],style=dict(width='32%')
                                    ),

                            dbc.Col([
                                    html.Div([
                                            "#Returned Orders by Order priority"
                                    ],style={'background-color':'#1D3537','font-family':'Segoe UI','color': 'white',
                                             'fontSize': 16, 'textAlign':'center'}
                                             ),
                                    dcc.Graph(id='histo2'),
                            ],style=dict(width='33%')
                                    ),
                            ]),
                    dbc.Row([
                            dbc.Col([
                                    html.Div([
                                            "% Returned Orders by Country"
                                    ] ,style={'background-color':'#1D3537','font-family':
                                              'Segoe UI','color': 'white',
                                              'fontSize': 16, 'textAlign':'center'}
                                             ),
                                    dcc.Graph(id='map1')
                            ],
                                ),
                            dbc.Col([
                                    dcc.Graph(id='table2'),
                
                                    html.Div(id='text2', style={'background-color':'#1D3537','font-family':
                                        'Segoe UI','color': 'white','fontSize': 16, 'textAlign':'left'})
                                            ], style=dict(width='35%',height='30%'),
                                    ),
                                    
                    ]),        
            ])

    
@app.callback(
    Output('sales_amount', 'children'),
    Output('returns_value', 'children'),
    Output('text_rate', 'children'),
    Output('donut1', 'figure'),
    Output('histo1', 'figure'),
    Output('tree1', 'figure'),
    Input('curr-check', 'value'),
    )

def call1(check):
    sales = df['Sales'].sum()
    returns_v = df_cust['Value'].sum(); 
    curr_text = ""
    sales_cur = sales
    returns_cur = returns_v
    cur = "USD"

    rate,curr,curr_text = rates(check)  # WORKS AS FUNCTION HERE 

    sales_cur = sales*float(rate)
    sales_amount = f"{sales_cur:,.2f}"
    returns_cur= returns_v*float(rate); 
    returns_value = f"{returns_cur:,.2f}"    
    
    donut = go.Figure(data=[go.Pie(labels=ship['Ship Mode'],values=ship['Count'], hole=.4)])
    donut.update_traces(hoverinfo='label+percent', textinfo='value',
                         textfont_size=16, marker=dict(line=dict(color='#000000', width=2)))
    donut.update_layout(width=400, height=400,margin={"pad": 0, "t": 0,"r": 150,"l": 0,"b": 0},
                        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.85))

    seg['Sum_rate'] = float(rate)*seg['Sum']
        
    histo =  px.histogram(x = seg['Segment'], y=seg['Sum_rate'],color_discrete_sequence=['#d64550'])
    histo.update_layout(margin={"pad": 0, "t": 50,"r": 100,"l": 0,"b": 0})
    histo.add_annotation(font=dict(color='black',size=14), x=2, y=seg['Sum_rate'].max(),showarrow=False,text=curr)
    histo.update_xaxes(showgrid=False,title="", tickfont=dict(size=14))
    histo.update_yaxes(showgrid=False, title="", tickfont=dict(size=14))

    tree = px.treemap(dft, path=[px.Constant(""), 'Order Priority'], values='Count',
                  color='Count',  color_continuous_scale='Rainbow', # https://plotly.com/python/builtin-colorscales/
                      )
    tree.update_layout(margin = dict(t=50, l=5, r=5, b=80))
  
    return '%s %s' %(sales_amount,curr),'%s %s' %(returns_value,curr),curr_text,donut,histo,tree


@app.callback(
    Output("cell-output", "children"),
    Output("table-month", "figure"),
    Input('curr-check', 'value'),
    Input("table", "active_cell"),
    State("table", "derived_viewport_data"),    
    )

def call2(check,cell,data):
    rate,curr,curr_text = rates(check) 
        
    if cell:
        selected = data[cell["row"]][cell["column_id"]]
        dfm = merge_month(selected); #print(dfm)
        if selected in list(df1234['Year']):
                cols = dfm.columns.values;
                table = go.Figure(data=[go.Table(
                        columnwidth = [100,60,100,60,100,60,100,60,100,60,100,60],
                        header=dict(values=[['Order Priority','Month'],
                             ['Critical','No. of Orders'],['','Sales Amount'],
                             ['High','No. of Orders'],['','Sales Amount'],
                             ['Low','No. of Orders'],['','Sales Amount'],
                             ['Medium','No. of Orders'],['','Sales Amount'],
                             ['Total','No. of Orders'],['','Sales Amount']
                             ],
                                    line_color='dimgrey',
                                    fill_color='dimgrey',font=dict(color='white', size=16),align='left',
                                    
                                    ),
                        cells=dict(values= [dfm['Month Name'],dfm['No orders'],dfm['Sales Amount'],
                             dfm['No orders_3'], dfm['Sales Amount_3'],
                             dfm['No orders_2'], dfm['Sales Amount_2'],
                             dfm['No orders_1'], dfm['Sales Amount_1'],
                             dfm['Orders total'], dfm['Sales total'],
                             ],
                                   fill_color='#E0EEEF', #https://htmlcolorcodes.com
                                   font=dict(color='black', size=16),
                                   height=30, align='left'))
                                        ])

                table.update_layout(margin={"pad": 0, "t": 0,"r": 0,"l": 0,"b": 0})
                return f"For {selected} (all USD)",table#,h2
        else:
                table = go.Figure(data=[go.Table()])
                return "Click year to expand and reload to compress", table#,h2
        
    else:
        table = go.Figure(data=[go.Table()])
        return "Click to expand and reload to compress", table#,h2       


@app.callback(
    Output('sales_amount2', 'children'),#,allow_duplicate=True),
    Output('returns_value2', 'children'),#,allow_duplicate=True),
    Output('text_rate2', 'children'),#,allow_duplicate=True),
    Output('donut2', 'figure'),
    Output('stack1', 'figure'),
    Output('histo2', 'figure'),
    Output('map1', 'figure'),
    Output('table2', 'figure'),
    Output('text2', 'children'),
    Input('curr-check', 'value'),
    #prevent_initial_call=True # NEED FOR DUPLICATE INPUTS - WORKS BUT WREAKS HAVOC ELSWEHERE
    )

def call3(check):

    sales = df['Sales'].sum()
    returns_v = df_cust['Value'].sum(); 
    curr_text = ""
    sales_cur = sales
    returns_cur = returns_v
    #curr = "USD"
    
    rate,curr,curr_text = rates(check);

    sales_cur = sales*float(rate)
    sales_amount = f"{sales_cur:,.2f}"
    returns_cur= returns_v*float(rate); 
    returns_value = f"{returns_cur:,.2f}"    

    donut2 = go.Figure(data=[go.Pie(labels=seg_ret['Segment'],values=seg_ret['Number'], hole=.4)])
    donut2.update_traces(hoverinfo='label+percent', textinfo='value',
                    textfont_size=16, marker=dict(line=dict(color='#000000', width=2)))
    donut2.update_layout(width=400, height=400,margin={"pad": 0, "t": 0,"r": 150,"l": 0,"b": 0},
                         legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.75))

    norm_US = 250 # WORKS WELL TO SCALE USD WITH HISTO
    norm = norm_US*float(rate)
    stack1 = px.line(x=mark_stack['Market'], y=float(rate)*mark_stack['Total']/norm,
                     color=px.Constant("Value of Returned Products"),
                     labels=dict(x="", y="",color=""))
    stack1.update_traces(line_color='#0000ff', line_width=5)
    stack1.add_bar(x = mark_stack['Market'], y=mark_stack['Number'], name="#Returned Orders")
    stack1.update_layout(legend=dict(yanchor="top",y=1.2,xanchor="left",x=0.01),
                         width=380, height=370,margin={"pad": 0, "t": 50,"r": 120,"l": 0,"b": 0})
    stack1.add_annotation(font=dict(color='black',size=14), x=3.5, y=1700 ,showarrow=False,text="%s/%1.0f" %(curr,norm))
    stack1.update_xaxes(showgrid=False,tickangle = 90)

    histo2 = go.Figure(go.Bar(y = pri_ret['Order Priority'], x=pri_ret['Number'],
                  orientation='h'))
    histo2.update_layout(width=380, height=320,margin={"pad": 0, "t": 20,"r": 100,"l": 20,"b": 0})
    histo2.update_xaxes(showgrid=False,tickangle = 90)

    #print(df_coun.head())
    df_coun['% returns'] = round(df_coun['perc'],2)
    
    world = px.choropleth(df_coun, locations = 'CODE',
              color_continuous_scale='Rainbow',
              color = '% returns',hover_name= "Country",)
    world.update_layout(width=550,height=450,
            margin={"pad": 0, "t": 0,"r": 0,"l": 0,"b": 0},
            geo=dict(showframe=False, showcoastlines=True),
            coloraxis_colorbar_title_text = '% '
            ),
    world.update_coloraxes(colorbar_orientation='h',colorbar_y=0)

    new_val = "Value [%s]" %(curr); 
    df_cust[new_val] = round(float(rate)*df_cust_orig['Value'],2) # A MESS IF NOT OVERWRITTEN
    #print(df_cust.head()); 
    cols = ['Customer Name','Customer ID','No returns', new_val]#df_cust.columns.values
    #print(cols)
    table = go.Figure(data=[go.Table(
            columnwidth = [10,13,8,8],
            header=dict(values=(cols),
            fill_color='#1D3537',
            font=dict(color='white', size=14),
            align='left'
            ),
           cells=dict(values= [df_cust['Customer Name'],df_cust['Customer ID'],df_cust['No returns'],df_cust[new_val]],
           fill_color='#E0EEEF', #https://htmlcolorcodes.com
           font=dict(color='black', size=14),
           height=30, align='left'),
    )
                        ])

    table.update_layout(margin={"pad": 0, "t": 0,"r": 0,"l": 0,"b": 0})

    tot_items = round(df_cust['No returns'].sum(),0)
    tot_sales = round(float(rate)*df_cust['Value'].sum(),0)
    
    return '%s %s' %(sales_amount,curr),'%s %s' %(returns_value,curr),curr_text,donut2,stack1,histo2,world,table,"Total .................................. %1.0f ... %1.0f" %(tot_items,tot_sales)
                      
if __name__ == '__main__':
    #app.run()  # pythonanywhere
    #app.run_server(debug=True) # http://Stevies-Air.staff.vuw.ac.nz:8050/ VUW
    #app.run_server(host = '172.20.10.3',debug=True) # http://172.20.10.3:8050/   PHONE
    app.run_server(host = '192.168.178.55',debug=True) #  http://Stevies-Air.fritz.box:8050/   HOME
