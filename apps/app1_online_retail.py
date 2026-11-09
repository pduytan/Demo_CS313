"""Application 1 — Online Retail Market Basket & Cross-Selling Pattern Explorer."""
from pathlib import Path
from html import escape
from collections import Counter
from itertools import combinations
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shared.demo_helpers import allowed_extensions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'app1_online_retail'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'app1_online_retail'

BAGS_KW = ('BAG', 'JUMBO', 'LUNCH', 'POUCH', 'PURSE', 'TOTE', 'SHOPPER', 'BACKPACK', 'SUKI')
KITCHEN_KW = ('TEACUP', 'SAUCER', 'MUG', 'CAKE', 'TIN', 'PLATE', 'BOWL', 'TEA', 'COFFEE', 'BOTTLE', 'GLASS', 'JAR', 'NAPKIN', 'SPOON', 'KITCHEN', 'BAKING')
DECOR_KW = ('HOLDER', 'T-LIGHT', 'HEART', 'CLOCK', 'FRAME', 'SIGN', 'BUNTING', 'LANTERN', 'CANDLE', 'LIGHT', 'BOX', 'HANGING', 'ORNAMENT', 'CUSHION', 'WALL', 'MIRROR')


def classify_product(item):
    _, name = item.split(' = ', 1)
    upper = name.upper()
    if any(k in upper for k in BAGS_KW):
        return 'Bags & storage'
    if any(k in upper for k in KITCHEN_KW):
        return 'Teacups, kitchen & tableware'
    if any(k in upper for k in DECOR_KW):
        return 'Home decor, holders & gifts'
    return 'Other products'


@st.cache_data
def load_retail():
    pairs = pd.read_csv(DATA_DIR / 'gio_hang_UK_da_xu_ly.csv', dtype=str)
    names = json.loads((DATA_DIR / 'ten_san_pham.json').read_text(encoding='utf-8'))
    baskets = pairs.groupby('InvoiceNo').StockCode.agg(list)
    n = len(baskets)
    total_products = int(pairs.StockCode.nunique())
    freq = Counter(x for row in baskets for x in row)
    codes = sorted(k for k, v in freq.items() if v / n >= 0.005)
    col_names = [f'{c} = {names.get(c, c)}' for c in codes]
    idx = {k: i for i, k in enumerate(codes)}
    arr = np.zeros((n, len(codes)), dtype=bool)
    for i, row in enumerate(baskets):
        for x in row:
            if x in idx:
                arr[i, idx[x]] = True
    matrix = pd.DataFrame(arr, index=baskets.index, columns=col_names)
    rules = pd.read_csv(RESULTS_DIR / 'association_rules.csv')
    rules['antecedents'] = rules.antecedent.map(lambda x: frozenset(x.split(' + ')))
    return matrix, rules, names, total_products


def label(item):
    return item


def describe(item):
    parts = item.split(' + ')
    formatted = []
    for p in parts:
        code, name = p.split(' = ', 1)
        formatted.append(f'product “{name}” ({code})')
    return 'include ' + ' and '.join(formatted)


def restore_selection(valid, previous):
    proposed = st.session_state.get('app1_items', [])
    if proposed and frozenset(proposed) not in valid:
        st.session_state.app1_items = list(previous)
        st.session_state._saved_app1_items = list(previous)
        st.session_state.app1_notice = 'This combination has no results under the current filters. Your previous selection was restored; clear all items to start again.'
    else:
        st.session_state._saved_app1_items = list(proposed)


def change_topic():
    st.session_state.app1_items = []
    st.session_state._saved_app1_items = []


def rule_diagram(selected, target, n, joint):
    chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in sorted(selected))
    target_chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in target.split(' + '))
    return (f'<div class="rule-flow"><div class="rule-node"><div class="rule-kicker">INPUT · ALL ITEMS PRESENT</div>'
            f'{chips}<div class="rule-note">{n:,} invoices match all input items</div></div>'
            '<div class="rule-arrow" aria-label="associated with">→</div>'
            f'<div class="rule-node rule-target"><div class="rule-kicker">ASSOCIATED PRODUCT(S)</div>'
            f'{target_chips}'
            f'<div class="rule-note">{joint:,}/{n:,} invoices also match this outcome · {joint/n:.1%}</div></div></div>'
            '<div class="rule-footnote">The arrow indicates the direction of the association rule, not causation.</div>')


def chart_note(text):
    st.markdown(f'<div class="chart-note"><strong>Observation</strong><p>{escape(text)}</p></div>', unsafe_allow_html=True)


def result_table(rows):
    body = ''.join(
        f'<tr><td>{escape(str(row.consequent))}</td><td>{int(row.joint_count):,}</td>'
        f'<td>{row.confidence:.2%}</td><td>{row.lift:.2f}×</td></tr>'
        for row in rows.head(20).itertuples())
    return ('<div class="results-table-wrap" role="region" aria-label="Associated responses" tabindex="0">'
            '<table class="results-table"><thead><tr><th scope="col">Response</th>'
            '<th scope="col">Joint count</th><th scope="col">Confidence</th>'
            '<th scope="col">Lift</th></tr></thead><tbody>' + body + '</tbody></table></div>')


def distribution_note(matrix, matched, target, peer_items):
    overall = matrix[peer_items].mean()
    group = matrix.loc[matched, peer_items].mean()
    top_item = group.idxmax()
    changes = group - overall
    up = changes.idxmax()
    lines = [
        f'Within the selected invoice group, “{top_item}” has the highest co-occurrence rate ({group.max():.1%}).',
        f'Compared with all {len(matrix):,} UK invoices, “{up}” shows the largest increase (+{100*changes[up]:.1f} percentage points).',
        f'The current rule focuses on “{target}”; these co-occurrence rates reflect invoice baskets, not causal purchase behavior.'
    ]
    return ' '.join(lines)


def query_rules(matrix, selected, support, confidence, lift):
    cols = ['antecedent', 'consequent', 'antecedents', 'antecedent_count', 'joint_count', 'support', 'confidence', 'baseline', 'lift']
    if not selected:
        return pd.DataFrame(columns=cols)
    matched = matrix[list(selected)].all(axis=1)
    n = int(matched.sum())
    if not n:
        return pd.DataFrame(columns=cols)
    joint = matrix.loc[matched].sum()
    base = matrix.mean()
    rows = []
    for item in matrix:
        if item in selected or not joint[item] or not base[item]:
            continue
        s = float(joint[item]) / len(matrix)
        c = float(joint[item]) / n
        l = c / base[item]
        if s >= support - 1e-12 and c >= confidence - 1e-12 and l > lift:
            rows.append(dict(
                antecedent=' + '.join(sorted(selected)),
                consequent=item,
                antecedents=selected,
                antecedent_count=n,
                joint_count=int(joint[item]),
                support=s,
                confidence=c,
                baseline=float(base[item]),
                lift=l,
            ))
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(show_spinner=False)
def query_navigation(matrix, selected, support, confidence, lift):
    valid = set()
    if selected and not query_rules(matrix, selected, support, confidence, lift).empty:
        valid.add(selected)
    for x in selected:
        sub = selected - {x}
        if sub and not query_rules(matrix, sub, support, confidence, lift).empty:
            valid.add(sub)
    if len(selected) >= 3:
        return valid
    mask = matrix[list(selected)].all(axis=1) if selected else pd.Series(True, index=matrix.index)
    values = matrix.loc[mask].to_numpy(dtype=np.float64)
    counts = values.sum(axis=0)
    joints = values.T @ values
    base = matrix.mean().to_numpy()
    conf = np.divide(joints, counts[:, None], out=np.zeros_like(joints), where=counts[:, None] > 0)
    lifts = np.divide(conf, base[None, :], out=np.zeros_like(conf), where=base[None, :] > 0)
    cols = list(matrix.columns)
    used_idx = {cols.index(x) for x in selected if x in cols}
    ok = (joints > 0) & (joints / len(matrix) >= support - 1e-12) & (conf >= confidence - 1e-12) & (lifts > lift)
    np.fill_diagonal(ok, False)
    for idx_u in used_idx:
        ok[:, idx_u] = False
    for i, item in enumerate(cols):
        if i not in used_idx and ok[i].any():
            valid.add(selected | {item})
    return valid


def expand_valid_sets(filtered):
    valid = set()
    for ant in filtered.antecedents:
        valid.add(ant)
        if len(ant) > 1:
            for r in range(1, len(ant)):
                for sub in combinations(ant, r):
                    valid.add(frozenset(sub))
    for con in filtered.consequent.unique():
        for part in con.split(' + '):
            valid.add(frozenset([part]))
    return valid


def main():
    matrix, rules, _, total_products = load_retail()
    st.markdown(f'''<div class="hero"><div class="eyebrow">UCI ONLINE RETAIL · UNITED KINGDOM</div>
    <h1>Discover Patterns in Market Basket &amp; Cross-Selling</h1>
    <p>Explore how retail products co-occur across customer invoices to support bundling and cross-selling.</p>
    <span class="badge">{len(matrix):,} valid UK invoices &nbsp; · &nbsp; {total_products:,} products</span></div>''', unsafe_allow_html=True)
    st.subheader('01 · Choose your starting features')
    with st.sidebar:
        st.subheader('Result filters')
        support = st.slider('Support', 0.0, 0.15, 0.02, 0.005, format='%.3f', key='app1_support')
        confidence = st.slider('Confidence', 0.0, 1.0, 0.60, 0.01, key='app1_confidence')
        lift = st.slider('Lift', 0.0, 20.0, 1.50, 0.10, key='app1_lift')
    filtered = rules[rules.support.ge(support - 1e-12) & rules.confidence.ge(confidence - 1e-12) & rules.lift.gt(lift)]
    live_query = support < 0.005 or confidence < 0.20 or lift < 1.20
    valid = expand_valid_sets(filtered)
    default_item = '22386 = JUMBO BAG PINK POLKADOT'
    default_list = [default_item] if default_item in matrix.columns else []
    if st.session_state.get('app1_navigation_version') != 1:
        st.session_state._saved_app1_items = list(default_list)
        st.session_state.app1_navigation_version = 1
    if 'app1_items' not in st.session_state:
        st.session_state.app1_items = list(st.session_state.get('_saved_app1_items', default_list))
    if 'app1_topic' not in st.session_state and '_saved_app1_topic' in st.session_state:
        st.session_state.app1_topic = st.session_state._saved_app1_topic
    topic = st.selectbox(
        'Browse features by topic',
        ['Bags & storage', 'Teacups, kitchen & tableware', 'Home decor, holders & gifts', 'All products'],
        key='app1_topic',
        on_change=change_topic,
    )
    st.session_state._saved_app1_topic = topic
    selected = frozenset(st.session_state.get('app1_items', []))
    if live_query:
        valid = query_navigation(matrix, selected, support, confidence, lift)
    if selected and selected not in valid:
        st.session_state.app1_items = []
        st.session_state._saved_app1_items = []
        selected = frozenset()
        if live_query:
            valid = query_navigation(matrix, selected, support, confidence, lift)
        st.info('Your previous selection no longer meets the filters. Start again with the available items below.')
    ordered = sorted(matrix.columns, key=lambda x: (classify_product(x) == 'Other products', label(x)))
    options = allowed_extensions(valid, selected, ordered)
    if topic != 'All products':
        options = [x for x in options if x in selected or classify_product(x) == topic]
    st.multiselect(
        'Select 1–3 product items',
        options,
        format_func=label,
        max_selections=3,
        key='app1_items',
        on_change=restore_selection,
        args=(valid, tuple(st.session_state.get('app1_items', []))),
        disabled=not options,
    )
    if st.session_state.get('app1_notice'):
        st.info(st.session_state.pop('app1_notice'))
    current_items = st.session_state.get('app1_items', [])
    st.session_state._saved_app1_items = list(current_items)
    selected = frozenset(current_items)
    if not selected:
        st.info('Select an item to get started.' if valid else 'No rules meet the current filters. Lower the thresholds to continue.')
        return
    matched = matrix[list(selected)].all(axis=1)
    n = int(matched.sum())
    candidates = filtered[filtered.antecedents.eq(selected)].sort_values(['confidence', 'lift'], ascending=False)
    if live_query:
        candidates = query_rules(matrix, selected, support, confidence, lift).sort_values(['confidence', 'lift'], ascending=False)
    candidates = candidates.copy()
    st.divider()
    st.subheader('02 · Explore associated responses')
    st.caption(f'The selected group contains {n:,} of {len(matrix):,} valid UK invoices.')
    if candidates.empty:
        ext_rules = filtered[filtered.antecedents.map(lambda a: selected.issubset(a) and len(a) > len(selected))]
        direct_fallback = rules[rules.antecedents.eq(selected)].sort_values(['confidence', 'lift'], ascending=False)
        if direct_fallback.empty:
            direct_fallback = query_rules(matrix, selected, 0.005, 0.10, 1.20).sort_values(['confidence', 'lift'], ascending=False)
        if not ext_rules.empty:
            companions = sorted({x for a in ext_rules.antecedents for x in (a - selected)})
            st.info(
                f'This selection participates in multi-item rules under the current filters (Confidence ≥ {confidence:.0%}). '
                f'Add a companion item above ({", ".join(companions[:3])}) to view the multi-item rule, or inspect its direct associations below.'
            )
        elif not direct_fallback.empty:
            st.info(
                f'As a standalone antecedent, this high-frequency product has a maximum confidence of {direct_fallback.confidence.max():.1%} '
                f'(below the current {confidence:.0%} filter, while appearing as a consequent or at lower confidence). '
                'Its direct associations are shown below.'
            )
        candidates = direct_fallback.copy()
    if candidates.empty:
        st.info('No results meet the current filters.')
        return
    rows = candidates.copy()
    rows['Response'] = rows.consequent.map(label)
    st.markdown(result_table(rows), unsafe_allow_html=True)
    result_options = list(candidates.consequent)
    initial_result = result_options[0]
    if st.session_state.get('app1_result') not in result_options:
        st.session_state.app1_result = initial_result
    chosen = st.selectbox(
        'Choose a response to examine',
        result_options,
        index=result_options.index(st.session_state.get('app1_result', initial_result)),
        format_func=label,
        key='app1_result',
    )
    row = candidates[candidates.consequent.eq(chosen)].iloc[0]
    target = row.consequent
    target_items = frozenset(target.split(' + '))
    joint = int(matrix[list(selected | target_items)].all(axis=1).sum())
    base = float(matrix[list(target_items)].all(axis=1).mean())
    conf = joint / n
    condition = ' and '.join(describe(x) for x in sorted(selected))
    outcome = describe(target)
    st.divider()
    st.subheader('03 · Understand the selected association')
    st.markdown(rule_diagram(selected, target, n, joint), unsafe_allow_html=True)
    st.markdown('**If an invoice were to ' + condition + '…**')
    st.markdown('**Associated response in this sample:** ' + outcome + '.')
    st.write(f'Of the **{n:,} invoices** matching all selected items, **{joint:,} ({conf:.1%})** also {outcome}.')
    direction = 'higher' if conf >= base else 'lower'
    st.write(f'In the full sample, this product occurs in **{base:.2%}** of invoices. The selected group is '
             f'**{100*abs(conf-base):.1f} percentage points {direction}**, or **{conf/base:.2f} times** the baseline rate.')
    st.info('Market basket rules reflect co-occurrence on valid UK invoices (excluding cancelled invoices and non-product fees), supporting product bundling and cross-selling.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Support', f'{joint/len(matrix):.2%}',
              help='Invoices matching both the input and outcome, divided by all valid UK invoices.')
    c1.caption(f'{joint:,}/{len(matrix):,} invoices in the full sample')
    c2.metric('Confidence', f'{conf:.2%}',
              help='Invoices matching both the input and outcome, divided by those matching the input.')
    c2.caption(f'{joint:,}/{n:,} invoices in the selected group')
    c3.metric('Lift', f'{conf/base:.2f}×',
              help='Confidence divided by the product frequency in the full sample.')
    c3.caption(f'{conf:.2%} / {base:.2%}')

    chart_context = ('app1', tuple(sorted(selected)), target, float(support), float(confidence), float(lift))
    if st.session_state.get('app1_chart_context') != chart_context:
        st.session_state.app1_chart_context = chart_context
        st.session_state.app1_show_charts = False
    if st.button('Hide analysis charts' if st.session_state.get('app1_show_charts') else 'Show analysis charts',
                 key='show_charts_button', icon=':material/monitoring:', type='primary'):
        st.session_state.app1_show_charts = not st.session_state.get('app1_show_charts', False)
        st.rerun()
    if st.session_state.get('app1_show_charts'):
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('**How common is this response?**')
            fig = px.bar(
                pd.DataFrame({'Group': ['Full sample', 'Selected group'], 'Proportion': [base, conf]}),
                x='Group', y='Proportion', color='Group',
                color_discrete_sequence=['#b7a4b1', '#3a0f34'], text_auto='.1%'
            )
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=410, margin=dict(t=30, b=25),
                yaxis=dict(range=[0, 1], tickformat='.0%')
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(f'Product “{target}” occurs in {conf:.1%} of the selected invoice group, '
                       f'versus {base:.2%} of the full sample: {100*abs(conf-base):.1f} percentage points {direction} '
                       f'(lift {conf/base:.2f}). The full sample includes the selected group.')
        with right:
            st.markdown('**Within the selected group**')
            fig = go.Figure(go.Pie(
                labels=['With target product', 'Without target product'],
                values=[joint, n - joint], hole=.65,
                marker_colors=['#3a0f34', '#d6cdd2'], textinfo='value+percent'
            ))
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=410, margin=dict(t=20, b=20), legend=dict(orientation='h'),
                annotations=[dict(text=f'{joint:,}/{n:,}', x=.5, y=.5, font_size=18, showarrow=False)]
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(f'Of {n:,} invoices matching all input items, {joint:,} also contain '
                       f'“{target}”; {n-joint:,} ({1-conf:.1%}) do not contain this product.')
        single_options = [x for c in result_options for x in c.split(' + ')]
        seen_peers = []
        for x in list(target_items) + single_options:
            if x in matrix.columns and x not in seen_peers:
                seen_peers.append(x)
        peer_items = seen_peers[:5]
        if len(peer_items) > 1:
            distributions = []
            for item in peer_items:
                short_code = item.split(' = ', 1)[0]
                distributions.extend([
                    {'Product code': short_code, 'Group': 'Full sample', 'Proportion': float(matrix[item].mean())},
                    {'Product code': short_code, 'Group': 'Selected group', 'Proportion': float(matrix.loc[matched, item].mean())},
                ])
            st.markdown('**How does the associated product profile compare?**')
            st.caption('Comparison of baseline vs. conditional presence across top associated products for this basket selection.')
            fig = px.bar(
                pd.DataFrame(distributions), x='Product code', y='Proportion', color='Group', barmode='group',
                color_discrete_map={'Full sample': '#b7a4b1', 'Selected group': '#3a0f34'}, text_auto='.1%'
            )
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=430, yaxis=dict(tickformat='.0%', range=[0, 1]), margin=dict(t=30, b=20)
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(distribution_note(matrix, matched, target, peer_items))
    st.caption('Co-occurrence in historical invoices does not imply causation or guaranteed purchase for every customer.')
    with st.expander('View itemset and detailed figures'):
        st.write('**Itemset:** ' + ' + '.join(label(x) for x in sorted(selected | target_items)))
        st.write('**Rule:** ' + ' + '.join(label(x) for x in sorted(selected)) + ' → ' + label(target))
        st.write(f'Support: {joint/len(matrix):.2%} · Confidence: {conf:.2%} · Lift: {conf/base:.2f}')
        st.write('Dataset: UCI Online Retail (United Kingdom valid invoices)')
        st.download_button(
            'Download results for this selection',
            rows[['antecedent', 'consequent', 'antecedent_count', 'joint_count', 'support', 'confidence', 'baseline', 'lift']].to_csv(index=False).encode('utf-8-sig'),
            'app1_online_retail_results.csv',
            'text/csv',
        )
