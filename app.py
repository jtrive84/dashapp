import os
import os.path
import sys
import re
import datetime
import time
import numpy as np
import pandas as pd
from scipy import stats
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table.FormatTemplate as FormatTemplate
import dash_table
from dash.dependencies import Input, Output

VALID_AUTH_PAIRS = {
    "fwa":"fwa",
    }


__file__ = "C:\\Users\\cac9159\\Repos\\LTC\\FWA\\Source\\App\\fwaapp.py"
add2path = os.path.dirname(__file__)
sys.path.append(add2path)
from prepdata import *

excl_            = ["ATTAINED_AGE", "POLICY_NUMBER", "CLAIM_NUMBER", "FRAUD_INDICATIOR",
                    "RESIDENT_STATE", "score", "priority", "DAILY_BENEFIT_INFL_BANDED",
                    "SITUS_CURRENT", "PREMIUM_PAYMENT_MODE",]

dfinput          = dfall
dfinput          = dfinput.drop("FRAUD_INDICATOR", axis=1) # imported from prepdata
claim_numbers    = sorted(dfinput["CLAIM_NUMBER"].unique().tolist())
dftypes          = dfinput.dtypes
dftypes          = dftypes[~dftypes.index.isin(excl_)].reset_index(drop=False)
dftypes.columns  = ["fieldname", "datatype"]
dftypes["plot"]  = dftypes["datatype"].map(lambda v: "hist" if v==np.float_ else "bar")
dftypes          = dftypes.sort_values("fieldname").reset_index(drop=True)
dftypes["varid"] = ["var" + "{}".format(i).zfill(2) for i in range(dftypes.shape[0])]



# Declarations ===============================================================]
def update_categorical(dfsubset, fieldname, claim_number, highlight_color="#CC0038",
                       other_color="#fa9fb5", highlight_alpha=1.0, other_alpha=.30,):
    """
    Refresh categorical variates based on claim_number.
    """
    fieldval_    = dfsubset.loc[0, fieldname]
    claim_color_ = highlight_color
    claim_alpha_ = highlight_alpha
    other_color_ = other_color
    other_alpha_ = other_alpha
    dfvar_       = dfinput[fieldname].value_counts().sort_index()
    xvals_       = dfvar_.index.values.tolist()
    yvals_       = dfvar_.values.tolist()
    bar_colors_  = [claim_color_ if i==fieldval_ else other_color_ for i in xvals_]
    bar_alphas_  = [claim_alpha_ if i==fieldval_ else other_alpha_ for i in xvals_]
    trace_ = go.Bar(
        x=xvals_, y=yvals_, text=yvals_, textposition="auto",
        marker={"color":bar_colors_, "opacity":bar_alphas_,
                "line":{"color":"#25232C", "width":0.35}}, showlegend=False,
        )
    layout_ = go.Layout(
        title=go.layout.Title(text=fieldname, x=.025, xref="paper", y=.85, yref="container"),
        xaxis={"showgrid":False, "showline":True}, yaxis={"showgrid":False, "showline":True, "showticklabels":False},
        margin=go.layout.Margin(l=60, r=60, t=75, b=75),
        )
    return({"data":[trace_], "layout":layout_})



# (lighter blue: #0eb9fe) (darker blue: #3a66b2) (default green: "#00CC94")
# (silver blue: #a5afd2) (aqua blue: 	#96c1d1) (lavender: 	#e8c7e8)
# (marker line color: blue: "#192d92"
def update_continuous(dfsubset, fieldname, claim_number, color="#beaadd", alpha=.8,
                      edge_color="#25232C", marker_line_color="#6834ab",
                      marker_line_style="line"):
    """
    Refresh continuous variates based on claim_number.
    """
    fieldval_ = dfsubset.loc[0, fieldname]
    xvals_    = dfinput[~np.isnan(dfinput[fieldname])][fieldname].values
    # nbinsx    = freedman_diaconis(data=xvals_, returnas="bins")
    # trace_ = go.Histogram(
    #     x=xvals_, nbinsx=nbinsx, histnorm="probability", opacity=hist_alpha,
    #     marker={"color":hist_color, "line":{"color":hist_edge_color, "width":0.5}},
    #     )

    # Compute maximum y-value to determine how to render vertical line
    # indicating the percentile of the continuous variable in relation to
    # the empirical distribution in question.
    n_, s_ = np.histogram(xvals_, bins="doane")
    m_ = n_.max()
    r_ = np.arange(0, m_ + 101, 50, dtype=np.int_)
    a_ = (r_ - m_)
    add_ = a_[a_>0].min()
    y1_ = (m_ + add_)
    size_ = s_[1] - s_[0]

    trace_ = go.Histogram(
        x=xvals_, opacity=alpha, histfunc="count", xbins={"size":size_}, histnorm="",
        marker={"color":color, "line":{"color":edge_color, "width":0.5}},
        )
    layout_ = go.Layout(
        title=go.layout.Title(text=fieldname, x=.025, xref="paper", y=.85, yref="container"),
        xaxis={"showgrid":True, "showline":True}, yaxis={"showgrid":True, "showline":True, "showticklabels":False},
        shapes=[{"type":"line", "x0":fieldval_, "y0":0, "x1":fieldval_, "y1":y1_,
                "line":{"color":marker_line_color, "width":3,}},],
        )

    return({"data":[trace_], "layout":layout_})








# Dashboard initialization ===================================================]
stylesheets_ = ["assets/css/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=stylesheets_)
auth = dash_auth.BasicAuth(app, VALID_AUTH_PAIRS)
app.config["suppress_callback_exceptions"] = True


# Layout =====================================================================]
layout_init = [
    html.Div([
        html.Div([
            html.H2("FWA Unsupervised Method Outlier Assessment"),
            ], style={"textAlign":"left", "width":"65%", "display":"inline-block"}
        ),
        html.Div([
            html.Img(src="assets/CNAmaster.jpg", style={"height":"70%", "width":"70%"}),
            ], style={"float":"right",  "width":"35%", "display":"inline-block", "padding":2}
        )
    ]),

    html.Div([
        dcc.RadioItems(
            id="which_subset",
            options=[
                {"label": "High Priority", "value": "high"},
                {"label": "Low Priority", "value": "low"},
                {"label": "All Claims", "value": "both"},
                ], value="both", labelStyle={"padding":2}),
        ], style={"padding":6}
    ),

    html.Div([
        html.Label("Select CLAIM_NUMBER:"),
        html.Div(dcc.Dropdown(
            id="claim_number", options=[{"label":i, "value":i} for i in claim_numbers],
            value="20001114", placeholder="CLAIM_NUMBER")),
        ], style={"width":250, "vertical-align":"middle", "padding":8},
    ),

    html.Div([
        html.Div(id="table")
        ], style={"width":"75%", "vertical-align":"middle", "padding":10},
    ),

    html.Div([
        dcc.Markdown(id="mdsumm")
        ], style={"width":"55%", "padding":10}
    ),
]

# Specify dynamic layout using 3 exhibits/plots per row ("four columns") =====]
layout_vars = []
excl_vars_  = ["cluster", "noise_id",]
varslist    = [i for i in dftypes.varid.values.tolist() if i not in excl_vars_]
nrows_      = len(varslist) // 3
final_      = len(varslist) % 3
head_       = list(range(0, len(varslist) + 1, 3))
tail_       = head_[1:] + [nrows_ * 3 + final_]
indx_       = list(zip(head_, tail_))

for i_1, i_2 in indx_:
    idlist = []
    for v_ in varslist[i_1:i_2]:
        varname_ = dftypes[dftypes.varid==v_]["fieldname"].values[0]
        # plot_ = dftypes[dftypes.varid==v_]["plot"].item()
        layout_ = {
            "xaxis" :{"showgrid":False, "showline":True},
            "yaxis" :{"showgrid":False, "showline":True, "showticklabels":False},
            "margin":go.layout.Margin(l=60, r=60, t=75, b=75),
            }
        spec_ = html.Div([
            dcc.Graph(id=v_, figure={"layout":layout_}),
            ], style={"width":"30%", "display":"inline-block"},
            className="four columns"
            )
        idlist.append(spec_)
    layout_vars.append(html.Div(idlist, className="row"))


# Hidden div to facilitate data update.
layout_final = [html.Div(id="intermediate", style={"display":"none"})]
# app.layout = html.Div(layout_init + layout_vars + layout_final)

l1 = "Claim numbers are assigned a priority score between 0-100. *Claims with scores\n"
l2 = "between **51-100** are of interest*, with higher scores representing those claims\n"
l3 = "with a relatively greater likelihood of questionable activity. Claims having\n"
l4 = "scores between **0-50** can be ignored."
priority_desc = l1 + l2 + l3 + l4



app.layout = html.Div([
    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label="Summary", children=[
            html.Div(layout_init + layout_vars + layout_final)
        ]),

        dcc.Tab(label="Priority Ranking", children=[

            html.Div([
                dcc.Markdown(
                    priority_desc
                    )], style={"width":"50%", "padding":25}
            ),

            html.Div([
                dash_table.DataTable(
                    # columns=[{"name": i, "id": i} for i in ["CLAIM_NUMBER", "POLICY_NUMBER", "priority"]],
                    # style_data={"border": "1px solid blue"},
                    # style_table={"overflowX":"scroll"},
                    # style_cell={'width': '150px'},
                    # style_data_conditional=[{"if": {"row_index": 4}, "backgroundColor": "#3D9970", 'color': 'white'}]
                    # fixed_rows={"headers":True, "data":0},
                    data=dfinput[["CLAIM_NUMBER", "POLICY_NUMBER", "priority"]].sort_values("priority", ascending=False).to_dict("records"),
                    columns=[
                        {"id":"CLAIM_NUMBER",  "name":"CLAIM_NUMBER",  "type":"text"},
                        {"id":"POLICY_NUMBER", "name":"POLICY_NUMBER", "type":"text"},
                        {"id":"priority",      "name":"priority",      "type":"numeric"},
                    ],
                    style_header={"border":"1px solid pink", "textAlign":"center"},
                    style_data_conditional=[
                        {"if": {"row_index":"odd"}, "backgroundColor":"rgb(248, 248, 248)"},
                    ],
                    style_cell_conditional=[
                        {"if": {"column_id": "CLAIM_NUMBER"}, "textAlign": "left",  "width":"10px", "maxWidth":"10px", "minWidth":"10px"},
                        {"if": {"column_id": "POLICY_NUMBER"},"textAlign": "left",  "width":"10px", "maxWidth":"10px", "minWidth":"10px"},
                        {"if": {"column_id": "priority"},     "textAlign": "right", "width":"10px", "maxWidth":"10px", "minWidth":"10px"},
                    ]
                )], style={"padding":25, "width":"50%", "margin-left":"25%",  "margin-right":"25%"}
            )
        ]),

        dcc.Tab(label="Map View", children=[
            html.Div([
                html.Label("Select Priority Threshold:"),
                html.Div(dcc.Dropdown(
                    id="threshold", options=[{"label":i, "value":i} for i in np.arange(50, 101, 1)],
                    value="90", placeholder="")),
                ], style={"width":200, "vertical-align":"middle", "padding":8},
            ),
            html.Div([
                dcc.Graph(id="mapping")
            ], style={"padding":0})
        ]),
    ])
])







# =============================================================================]
@app.callback(
    dash.dependencies.Output("claim_number", "options"),
    [dash.dependencies.Input("which_subset", "value")])
def update_claim_numbers(which_subset):
    """
    Update options available in dropdown based on selcted radioitem.
    """
    if which_subset=="both":
        claim_numbers_ = sorted(np.unique(dfinput["CLAIM_NUMBER"].values))
    elif which_subset=="high":
        claim_numbers_ = sorted(np.unique(dfinput[dfinput.priority>50].CLAIM_NUMBER.values))
    elif which_subset=="low":
        claim_numbers_ = sorted(np.unique(dfinput[dfinput.priority<=50].CLAIM_NUMBER.values))
    return([{"label":i, "value":i} for i in claim_numbers_])



@app.callback(
    dash.dependencies.Output("mdsumm", "children"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value")]
    )
def update_md(_json, claim_number):
    """
    Update markdown summary of given claim's values per attribute.
    """
    # claim_number = "4301064194"
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]==claim_number].reset_index(drop=True)
    
    dfsubset_ = pd.read_json(_json, orient="split").reset_index(drop=True)
    priority_ = dfsubset_.priority.values[0]

    md_str_ = "#### Summary for **{}** (priority score: {:.2f}):  \n```".format(claim_number, priority_)
    vardict = {}
    for indx_ in range(dftypes.shape[0]):

        var_ = dftypes.loc[indx_, "fieldname"]
        val_ = dfsubset_.loc[0, var_]
        plot_ = dftypes.loc[indx_, "plot"]

        if plot_=="bar":
            # Determine which proportion of variable equal var_.
            dfvar_ = dfinput[var_].value_counts().sort_index().reset_index(drop=False)
            dfvar_["ratio"] = dfvar_[var_] / dfvar_[var_].sum()
            prop_ = dfvar_[dfvar_["index"]==val_]["ratio"].values[0]
            summstr_ = "* Proportion of records with {} = {}: {:.3f}  \n".format(var_, val_, prop_)

        elif plot_=="hist":
            dfvar_ = dfinput[[var_]].sort_values(var_).reset_index(drop=True)
            dfvar_["cdf"] = dfvar_[var_] / dfvar_[var_].max()
            dfvar_["ecdf"] = (1 / dfvar_.shape[0])
            dfvar_["ecdf"] = np.cumsum(dfvar_["ecdf"])
            prop_ = dfvar_[dfvar_[var_]==val_]["ecdf"].values[0]
            summstr_ = "* Proportion of records with {} <= {}: {:.3f}  \n".format(var_, val_, prop_)
        vardict[prop_] = summstr_
    _, varstrs_ = zip(*sorted(vardict.items(), key=lambda v: abs(.5 - v[0]), reverse=True))
    for mds_ in varstrs_: md_str_+=mds_
    md_str_+="```"
    return(md_str_)





@app.callback(
    dash.dependencies.Output("intermediate", "children"),
    [dash.dependencies.Input("claim_number", "value")]
    )
def filter_data(claim_number):
    """
    Filter dfinput to selected claim_number.
    """
    dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]==claim_number].reset_index(drop=True)
    return(dfsubset_.to_json(orient="split"))



@app.callback(
    dash.dependencies.Output("table", "children"),
    [dash.dependencies.Input("intermediate", "children")]
    )
def update_table(_json):
    """
    Update rendered HTML table. Uses dash_table.DataTable.
    """
    dfsubset_ = pd.read_json(_json, orient="split").reset_index(drop=True)
    data_ = dfsubset_.to_dict("records")
    columns_ = [{"name": i, "id": i,} for i in dfsubset_.columns.values]
    table_ = dash_table.DataTable(
        data=data_, columns=columns_, style_table={"overflowX":"scroll"},
        # style_cell={"minWidth":"0px", "maxWidth":"250px",},
        )
    return(table_)



@app.callback(
    dash.dependencies.Output("mapping", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("threshold", "value")]
    )
def update_map(_json, threshold):
    """
    Update rendered dash_table.DataTable.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_ = pd.read_json(_json, orient="split").reset_index(drop=True)
    dfmap_ = dfinput[dfinput.priority>=int(threshold)].reset_index(drop=True)
    dfmap_["count"] = 1
    dfmap_["avg_priority"] = dfmap_.groupby("RESIDENT_STATE", as_index=False)["priority"].transform("mean")
    keepcols_ = ["RESIDENT_STATE", "count", "avg_priority"]
    dfmap_ = dfmap_[keepcols_].groupby(["RESIDENT_STATE", "avg_priority"], as_index=False).sum()
    dfmap_["text"] = dfmap_["RESIDENT_STATE"] + "<br>" + " Count " + dfmap_["count"].astype(np.str) + \
                     "<br>" + " Average Priority " + dfmap_["avg_priority"].astype(np.str)
    scl = [
        [0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'], [0.4, 'rgb(188,189,220)'],
        [0.6, 'rgb(158,154,200)'], [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']
        ]

    trace_ = go.Choropleth(
        colorscale=scl,
        autocolorscale=False,
        locations=dfmap_["RESIDENT_STATE"],
        z=dfmap_["count"],
        locationmode="USA-states",
        text=dfmap_["text"],
        marker=go.choropleth.Marker(line=go.choropleth.marker.Line(color="rgb(255,255,255)", width=1)),
        # colorbar=go.choropleth.ColorBar(title="Frequency"),
        )

    layout_ = go.Layout(
        title=go.layout.Title(text="Distribution of Priority Scores by State"),
        geo = go.layout.Geo(
            scope="usa",
            projection=go.layout.geo.Projection(type="albers usa"),
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
            ),
        )

    return({"data":[trace_], "layout":layout_})



@app.callback(
    dash.dependencies.Output("var00", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var00(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[0, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[0, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var01", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var01(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[1, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[1, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var02", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var02(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[2, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[2, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var03", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var03(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[3, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[3, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var04", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var04(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[4, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[4, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var05", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var05(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[5, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[5, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var06", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var06(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[6, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[6, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var07", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var07(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[7, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[7, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)




@app.callback(
    dash.dependencies.Output("var08", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var08(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[8, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[8, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var09", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var09(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[9, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[9, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var10", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var10(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[10, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[10, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var11", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var11(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[11, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[11, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)


@app.callback(
    dash.dependencies.Output("var12", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var12(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[12, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[12, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)


@app.callback(
    dash.dependencies.Output("var13", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var13(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[13, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[13, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var14", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var14(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[14, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[14, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)



@app.callback(
    dash.dependencies.Output("var15", "figure"),
    [dash.dependencies.Input("intermediate", "children"),
     dash.dependencies.Input("claim_number", "value"),]
    )
def update_var15(_json, claim_number):
    """
    Update visualizations based on attributes of selected CLAIM_NUMBER.
    Note that a figure represents a dictionary containing two keys:
    {"data":trace, "layout":layout}.
    """
    # dfsubset_ = dfinput[dfinput["CLAIM_NUMBER"]=="4301064194"].reset_index(drop=True)
    dfsubset_  = pd.read_json(_json, orient="split")
    fieldname_ = dftypes.loc[15, "fieldname"]
    fieldval_  = dfsubset_.loc[0, fieldname_]
    plot_      = dftypes.loc[15, "plot"]

    if plot_=="bar":
        dexhibit_ = update_categorical(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )
    elif plot_=="hist":
        dexhibit_ = update_continuous(
            dfsubset=dfsubset_, fieldname=fieldname_, claim_number=claim_number,
            )

    return(dexhibit_)






if __name__ == "__main__":

    app.run_server(port=9999, debug=True, threaded=False)