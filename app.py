"""Unified FP-Growth Pattern Explorer — 3 Applications in 1 Interface."""
import streamlit as st
from apps.app1_online_retail import main as app1_main
from apps.app2_young_people_survey import main as app2_main
from apps.app3_mimic_iv import main as app3_main

st.set_page_config(
    page_title='FP-Growth Association Rule Explorer',
    page_icon='🧩',
    layout='wide',
)

SHARED_CSS = '''<style>
html{font-size:16px}
:root{--ink:#3a0f34;--muted:#73616f;--accent:#3a0f34;--line:#d6cdd2}
/* Dropdown menus render in a portal, outside the main content container. */
[data-baseweb="select"], [data-baseweb="select"] input,
[data-baseweb="select"] [data-baseweb="tag"],
[data-baseweb="select"] [data-baseweb="tag"] span,
[data-baseweb="select"] [data-baseweb="tag"] [role="presentation"],
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="menu"] li, [data-baseweb="menu"] li *,
[role="listbox"] [role="option"], [role="listbox"] [role="option"] * {
    font-size:18px!important;line-height:1.5!important;
}
[data-baseweb="tag"]{height:auto!important;min-height:38px;padding:5px 10px!important}
[data-baseweb="popover"] li,[role="option"]{min-height:48px}
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stMultiSelect"] [data-baseweb="select"] div {
    font-size:18px!important;
}
.stApp{background:#f5f2ef;color:var(--ink);color-scheme:light}
.results-table-wrap{overflow:auto;max-height:620px;border:1px solid #d6cdd2;border-radius:14px;background:#fffdfa;margin:8px 0 24px}
table.results-table{width:100%;border-collapse:separate;border-spacing:0;font-size:18px!important;line-height:1.5;table-layout:auto}
table.results-table th,table.results-table td{font-size:18px!important;line-height:1.5!important;padding:16px 20px;border:0;border-bottom:1px solid #e6dfe3;color:#3a0f34;vertical-align:middle}
table.results-table th{position:sticky;top:0;background:#eee7eb;font-weight:600;text-align:right;white-space:nowrap;z-index:1}
table.results-table td{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.results-table th:first-child,table.results-table td:first-child{text-align:left;white-space:normal;min-width:280px;width:48%;overflow-wrap:anywhere}
table.results-table tr:last-child td{border-bottom:0}
table.results-table tbody tr:hover{background:#f5eff3}
.results-table-wrap:focus-visible{outline:2px solid #612551;outline-offset:3px}
.stMainBlockContainer{max-width:1440px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3{font-size:18px!important;line-height:1.65!important;font-weight:700!important;color:var(--ink)}
[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p,[data-baseweb="select"],input,button{font-size:18px!important;line-height:1.65}
[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"],[data-testid="stMetricLabel"],
[data-testid="stMetricValue"]{color:var(--ink)}
[data-testid="stCaptionContainer"] p{color:var(--muted)}
.hero{background:linear-gradient(112deg,#3a0f34 0%,#3a0f34 72%,#663555 72%,#663555 100%);border-radius:20px;padding:28px 32px;margin-bottom:22px;border:1px solid #633653;box-shadow:0 10px 28px #3a0f3412}
.hero .eyebrow{color:#eaaa00;font-weight:600;letter-spacing:.08em;margin-bottom:12px}
.hero h1{font-size:18px!important;color:#fff!important;margin:0 0 10px!important;padding:0!important}
.hero p{color:#f5e8ef!important;margin:0;max-width:800px}
.hero .badge{display:inline-block;margin-top:20px;color:#eaaa00;background:#ffffff12;border:1px solid #ffffff38;border-radius:8px;padding:5px 12px}
[data-testid="stSidebar"]{background:#e5e4e2;border-right:1px solid #d2cbcf}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{color:#3a0f34}
[data-testid="stSidebar"] h3{color:#612551!important}
[data-testid="stExpander"]{background:#fffdfa;border:1px solid var(--line);border-radius:12px;box-shadow:0 2px 6px #3a0f3403}
[data-testid="stMetric"]{background:#fffdfa;padding:24px 26px;border-radius:14px;border:1px solid var(--line);border-top:3px solid #eaaa00;box-shadow:0 4px 12px #3a0f3406;min-height:106px}
[data-testid="stMetricLabel"] p{font-size:18px!important;line-height:1.65!important;color:#73616f}
[data-testid="stMetricValue"],[data-testid="stMetricValue"]>div{font-size:18px!important;line-height:1.65!important;font-weight:700;color:#612551}
[data-baseweb="select"]>div{background:#fffdfa;color:var(--ink);border-color:#cbbfc6;border-radius:10px;min-height:56px}
[data-baseweb="tag"]{background:#efe1e8;color:#612551;border:1px solid #d2b9c8;border-radius:6px}
[data-baseweb="tag"] span{color:#612551}
[role="option"],[role="listbox"]{background:#fffdfa;color:var(--ink)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stPlotlyChart"]{background:#fffdfa;border:1px solid var(--line);border-radius:14px;padding:8px;box-shadow:0 4px 12px #3a0f3404}
[data-testid="stAlert"]{border-radius:12px}
[data-testid="stDataFrame"]{--gdg-base-font-style:18px;--gdg-header-font-style:600 18px;--gdg-cell-horizontal-padding:16px}
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"]{font-size:18px!important}

button[kind="secondary"]{border-radius:9px;border-color:#cbbfc6}
.rule-flow{display:grid;grid-template-columns:minmax(0,1fr) 38px minmax(0,1fr);align-items:center;gap:12px;margin:16px 0 8px}
.rule-node{background:#fffdfa;border:1px solid #d6cdd2;border-radius:14px;padding:20px;min-width:0}
.rule-target{background:#eadde5;border-color:#c6a6bb}
.rule-kicker{font-size:18px;font-weight:700;color:#612551;margin-bottom:12px}
.rule-chip{display:block;border:1px solid #d8cbd3;background:#f9f4f6;border-radius:8px;padding:8px 12px;margin:6px 0;overflow-wrap:anywhere;font-size:18px}
.rule-arrow{font-size:18px;text-align:center;font-weight:700;color:#eaaa00}
.rule-note,.rule-footnote{font-size:18px;color:#73616f;line-height:1.65;margin-top:12px}
.rule-footnote{margin-bottom:20px}
.chart-note{background:#eee6e9;border-left:3px solid #b7a4b1;border-radius:0 10px 10px 0;padding:14px 18px;margin:0 0 20px;line-height:1.65}
.chart-note strong{color:#612551;font-size:18px}
.chart-note p{margin:5px 0 0;font-size:18px;color:#523c4d}
.st-key-show_charts_button button{background:#eaaa00!important;color:#3a0f34!important;border:1px solid #eaaa00!important;border-radius:12px!important;padding:12px 22px!important;min-height:58px;font-weight:700;box-shadow:0 4px 20px #eaaa0026;transition:transform .15s,box-shadow .15s}
.st-key-show_charts_button button:hover{background:#f5bd2b!important;box-shadow:0 6px 26px #eaaa0038;transform:translateY(-1px)}
.st-key-show_charts_button button p,.st-key-show_charts_button button span{color:#3a0f34!important}
.st-key-show_charts_button button:focus-visible{outline:2px solid #3a0f34;outline-offset:4px}
@media(max-width:700px){.rule-flow{grid-template-columns:1fr}.rule-arrow{transform:rotate(90deg)}}
@media(max-width:700px){.stMainBlockContainer{padding:1.2rem}.hero{padding:22px}}
</style>'''

st.markdown(SHARED_CSS, unsafe_allow_html=True)

APPS = {
    'Application 1 · Market Basket & Cross-Selling (Online Retail)': app1_main,
    'Application 2 · Personality & Behavior (Young People Survey)': app2_main,
    'Application 3 · Clinical Comorbidities (MIMIC-IV)': app3_main,
}

with st.sidebar:
    st.subheader('Select application')
    selected_app = st.selectbox(
        'Application',
        list(APPS.keys()),
        index=1,
        key='active_application',
        label_visibility='collapsed',
    )
    st.divider()

APPS[selected_app]()
