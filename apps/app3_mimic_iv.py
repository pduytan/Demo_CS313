"""Application 3 — Clinical Comorbidity Pattern Explorer (MIMIC-IV Clinical Database Demo 2.2)."""
from pathlib import Path
from html import escape
from itertools import combinations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shared.demo_helpers import allowed_extensions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'app3_mimic_iv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'app3_mimic_iv'
NOT_DISEASE = r'^(?:9_[VE]|10_[RVWXYZ])'

CARDIOMETABOLIC_PREFIXES = (
    '9_250', '9_272', '9_276', '9_278', '9_401', '9_402', '9_403', '9_404', '9_405',
    '9_410', '9_411', '9_412', '9_413', '9_414', '9_415', '9_416', '9_424', '9_425',
    '9_426', '9_427', '9_428', '9_429', '9_433', '9_434', '9_437', '9_438', '9_440',
    '9_441', '9_443', '10_E', '10_I',
)
RENAL_RESP_HEME_PREFIXES = (
    '9_280', '9_281', '9_282', '9_283', '9_284', '9_285', '9_286', '9_287', '9_288', '9_289',
    '9_46', '9_47', '9_48', '9_49', '9_50', '9_51', '9_58', '9_59',
    '10_D5', '10_D6', '10_D7', '10_D8', '10_J', '10_N',
)


def classify_diagnosis(item):
    code, _ = item.split(' = ', 1)
    if any(code.startswith(p) for p in CARDIOMETABOLIC_PREFIXES):
        return 'Cardiometabolic & endocrine'
    if any(code.startswith(p) for p in RENAL_RESP_HEME_PREFIXES):
        return 'Renal, respiratory & hematologic'
    return 'Gastrointestinal, neurologic & other'


@st.cache_data
def load_mimic():
    dx = pd.read_csv(DATA_DIR / 'hosp' / 'diagnoses_icd.csv.gz', dtype={'icd_code': str})
    d_icd = pd.read_csv(DATA_DIR / 'hosp' / 'd_icd_diagnoses.csv.gz', dtype={'icd_code': str})
    for t in (dx, d_icd):
        t['icd_code'] = t['icd_code'].str.strip()
        t['icd3'] = t['icd_version'].astype(str) + '_' + t['icd_code'].str[:3]
    dx = dx[~dx['icd3'].str.contains(NOT_DISEASE, regex=True)].copy()
    top = (
        dx.groupby(['icd3', 'icd_version', 'icd_code'])
        .size()
        .reset_index(name='n')
        .sort_values('n', ascending=False)
        .drop_duplicates('icd3')
        .merge(d_icd[['icd_code', 'icd_version', 'long_title']], on=['icd_code', 'icd_version'], how='left')
    )
    icd_names = top.set_index('icd3')['long_title'].to_dict()
    patient_groups = dx.groupby('subject_id')['icd3'].apply(lambda s: sorted(set(s)))
    all_codes = sorted(dx['icd3'].unique())
    col_names = [f'{c} = {icd_names.get(c, c)}' for c in all_codes]
    idx = {c: i for i, c in enumerate(all_codes)}
    arr = np.zeros((len(patient_groups), len(all_codes)), dtype=bool)
    for i, codes in enumerate(patient_groups):
        for c in codes:
            arr[i, idx[c]] = True
    matrix = pd.DataFrame(arr, index=patient_groups.index, columns=col_names)
    rules = pd.read_csv(RESULTS_DIR / 'association_rules.csv')
    rules['antecedents'] = rules.antecedent.map(lambda x: frozenset(x.split(' + ')))
    return matrix, rules, icd_names, len(all_codes)


def label(item):
    return item


def describe(item):
    parts = item.split(' + ')
    formatted = []
    for p in parts:
        code, title = p.split(' = ', 1)
        formatted.append(f'diagnosis group “{title}” ({code})')
    return 'have ' + ' and '.join(formatted)


def restore_selection(valid, previous):
    proposed = st.session_state.get('app3_items', [])
    if proposed and frozenset(proposed) not in valid:
        st.session_state.app3_items = list(previous)
        st.session_state._saved_app3_items = list(previous)
        st.session_state.app3_notice = 'This combination has no results under the current filters. Your previous selection was restored; clear all items to start again.'
    else:
        st.session_state._saved_app3_items = list(proposed)


def change_topic():
    st.session_state.app3_items = []
    st.session_state._saved_app3_items = []


def rule_diagram(selected, target, n, joint):
    chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in sorted(selected))
    target_chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in target.split(' + '))
    return (f'<div class="rule-flow"><div class="rule-node"><div class="rule-kicker">INPUT · ALL ITEMS PRESENT</div>'
            f'{chips}<div class="rule-note">{n} patients match all input diagnoses</div></div>'
            '<div class="rule-arrow" aria-label="associated with">→</div>'
            f'<div class="rule-node rule-target"><div class="rule-kicker">ASSOCIATED DIAGNOSIS</div>'
            f'{target_chips}'
            f'<div class="rule-note">{joint}/{n} patients also match this outcome · {joint/n:.1%}</div></div></div>'
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
        f'Within the selected patient subgroup, “{top_item}” has the highest prevalence ({group.max():.1%}).',
        f'Compared with all {len(matrix)} patients, “{up}” shows the largest increase (+{100*changes[up]:.1f} percentage points).',
        f'The current rule concerns “{target}”; these associations reflect clinical comorbidity in the hospital cohort, not direct disease causation.'
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
    matrix, rules, _, total_icd3 = load_mimic()
    st.markdown(f'''<div class="hero"><div class="eyebrow">MIMIC-IV CLINICAL DATABASE DEMO 2.2</div>
    <h1>Discover Patterns in Clinical Comorbidities</h1>
    <p>Explore how 3-character ICD-9 and ICD-10 disease categories co-occur across hospital patient records.</p>
    <span class="badge">{len(matrix)} patients &nbsp; · &nbsp; {total_icd3} ICD-3 disease groups</span></div>''', unsafe_allow_html=True)
    st.subheader('01 · Choose your starting features')
    with st.sidebar:
        st.subheader('Result filters')
        support = st.slider('Support', 0.0, 0.50, 0.10, 0.01, key='app3_support')
        confidence = st.slider('Confidence', 0.0, 1.0, 0.60, 0.01, key='app3_confidence')
        lift = st.slider('Lift', 0.0, 6.0, 1.20, 0.05, key='app3_lift')
    filtered = rules[rules.support.ge(support - 1e-12) & rules.confidence.ge(confidence - 1e-12) & rules.lift.gt(lift)]
    live_query = support < 0.05 or confidence < 0.25 or lift < 1.00
    valid = expand_valid_sets(filtered)
    default_item = '9_272 = Other and unspecified hyperlipidemia'
    default_list = [default_item] if default_item in matrix.columns else []
    if st.session_state.get('app3_navigation_version') != 1:
        st.session_state._saved_app3_items = list(default_list)
        st.session_state.app3_navigation_version = 1
    if 'app3_items' not in st.session_state:
        st.session_state.app3_items = list(st.session_state.get('_saved_app3_items', default_list))
    if 'app3_topic' not in st.session_state and '_saved_app3_topic' in st.session_state:
        st.session_state.app3_topic = st.session_state._saved_app3_topic
    topic = st.selectbox(
        'Browse features by topic',
        ['Cardiometabolic & endocrine', 'Renal, respiratory & hematologic', 'Gastrointestinal, neurologic & other', 'All diagnoses'],
        key='app3_topic',
        on_change=change_topic,
    )
    st.session_state._saved_app3_topic = topic
    selected = frozenset(st.session_state.get('app3_items', []))
    if live_query:
        valid = query_navigation(matrix, selected, support, confidence, lift)
    if selected and selected not in valid:
        st.session_state.app3_items = []
        st.session_state._saved_app3_items = []
        selected = frozenset()
        if live_query:
            valid = query_navigation(matrix, selected, support, confidence, lift)
        st.info('Your previous selection no longer meets the filters. Start again with the available items below.')
    ordered = sorted(matrix.columns, key=lambda x: (classify_diagnosis(x) == 'Gastrointestinal, neurologic & other', label(x)))
    options = allowed_extensions(valid, selected, ordered)
    if topic != 'All diagnoses':
        options = [x for x in options if x in selected or classify_diagnosis(x) == topic]
    st.multiselect(
        'Select 1–3 diagnosis items',
        options,
        format_func=label,
        max_selections=3,
        key='app3_items',
        on_change=restore_selection,
        args=(valid, tuple(st.session_state.get('app3_items', []))),
        disabled=not options,
    )
    if st.session_state.get('app3_notice'):
        st.info(st.session_state.pop('app3_notice'))
    current_items = st.session_state.get('app3_items', [])
    st.session_state._saved_app3_items = list(current_items)
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
    st.caption(f'The selected group contains {n} of {len(matrix)} patients.')
    if candidates.empty:
        ext_rules = filtered[filtered.antecedents.map(lambda a: selected.issubset(a) and len(a) > len(selected))]
        direct_fallback = rules[rules.antecedents.eq(selected)].sort_values(['confidence', 'lift'], ascending=False)
        if direct_fallback.empty:
            direct_fallback = query_rules(matrix, selected, 0.05, 0.20, 1.00).sort_values(['confidence', 'lift'], ascending=False)
        if not ext_rules.empty:
            companions = sorted({x for a in ext_rules.antecedents for x in (a - selected)})
            st.info(
                f'This selection participates in multi-item rules under the current filters (Confidence ≥ {confidence:.0%}). '
                f'Add a companion diagnosis above ({", ".join(companions[:3])}) to view the multi-item rule, or inspect its direct associations below.'
            )
        elif not direct_fallback.empty:
            st.info(
                f'As a standalone antecedent, this diagnosis has a maximum confidence of {direct_fallback.confidence.max():.1%} '
                f'(below the current {confidence:.0%} filter). Its direct associations are shown below.'
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
    if st.session_state.get('app3_result') not in result_options:
        st.session_state.app3_result = initial_result
    chosen = st.selectbox(
        'Choose a response to examine',
        result_options,
        index=result_options.index(st.session_state.get('app3_result', initial_result)),
        format_func=label,
        key='app3_result',
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
    st.markdown('**If a patient record were to ' + condition + '…**')
    st.markdown('**Associated response in this sample:** ' + outcome + '.')
    st.write(f'Of the **{n} patients** matching all selected diagnoses, **{joint} ({conf:.1%})** also {outcome}.')
    direction = 'higher' if conf >= base else 'lower'
    st.write(f'In the full sample, this diagnosis occurs in **{base:.1%}** of patients. The selected group is '
             f'**{100*abs(conf-base):.1f} percentage points {direction}**, or **{conf/base:.2f} times** the baseline rate.')
    st.info('Diagnoses are grouped by 3-character ICD-9/ICD-10 prefix (excluding non-disease V/E/R/W/X/Y/Z codes) and named after the most frequent full ICD code in each group.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Support', f'{joint/len(matrix):.2%}',
              help='Patients matching both the input and outcome diagnoses, divided by the full cohort.')
    c1.caption(f'{joint}/{len(matrix)} patients in the full sample')
    c2.metric('Confidence', f'{conf:.2%}',
              help='Patients matching both the input and outcome diagnoses, divided by those matching the input.')
    c2.caption(f'{joint}/{n} patients in the selected group')
    c3.metric('Lift', f'{conf/base:.2f}×',
              help='Confidence divided by the diagnosis frequency in the full cohort.')
    c3.caption(f'{conf:.2%} / {base:.2%}')

    chart_context = ('app3', tuple(sorted(selected)), target, float(support), float(confidence), float(lift))
    if st.session_state.get('app3_chart_context') != chart_context:
        st.session_state.app3_chart_context = chart_context
        st.session_state.app3_show_charts = False
    if st.button('Hide analysis charts' if st.session_state.get('app3_show_charts') else 'Show analysis charts',
                 key='show_charts_button', icon=':material/monitoring:', type='primary'):
        st.session_state.app3_show_charts = not st.session_state.get('app3_show_charts', False)
        st.rerun()
    if st.session_state.get('app3_show_charts'):
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
            chart_note(f'Diagnosis “{target}” occurs in {conf:.1%} of the selected patient group, '
                       f'versus {base:.1%} of the full sample: {100*abs(conf-base):.1f} percentage points {direction} '
                       f'(lift {conf/base:.2f}). The full sample includes the selected group.')
        with right:
            st.markdown('**Within the selected group**')
            fig = go.Figure(go.Pie(
                labels=['With target diagnosis', 'Without target diagnosis'],
                values=[joint, n - joint], hole=.65,
                marker_colors=['#3a0f34', '#d6cdd2'], textinfo='value+percent'
            ))
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=410, margin=dict(t=20, b=20), legend=dict(orientation='h'),
                annotations=[dict(text=f'{joint}/{n}', x=.5, y=.5, font_size=18, showarrow=False)]
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(f'Of {n} patients matching all input diagnoses, {joint} also have '
                       f'“{target}”; {n-joint} ({1-conf:.1%}) do not have this diagnosis recorded.')
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
                    {'ICD-3 group': short_code, 'Group': 'Full sample', 'Proportion': float(matrix[item].mean())},
                    {'ICD-3 group': short_code, 'Group': 'Selected group', 'Proportion': float(matrix.loc[matched, item].mean())},
                ])
            st.markdown('**How does the comorbidity profile compare?**')
            st.caption('Comparison of baseline vs. conditional prevalence across top associated ICD-3 diagnosis groups.')
            fig = px.bar(
                pd.DataFrame(distributions), x='ICD-3 group', y='Proportion', color='Group', barmode='group',
                color_discrete_map={'Full sample': '#b7a4b1', 'Selected group': '#3a0f34'}, text_auto='.1%'
            )
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=430, yaxis=dict(tickformat='.0%', range=[0, 1]), margin=dict(t=30, b=20)
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(distribution_note(matrix, matched, target, peer_items))
    st.caption('Clinical co-occurrence in hospital discharge codes reflects comorbidity patterns in this sample, not causal progression or diagnostic certainty.')
    with st.expander('View itemset and detailed figures'):
        st.write('**Itemset:** ' + ' + '.join(label(x) for x in sorted(selected | target_items)))
        st.write('**Rule:** ' + ' + '.join(label(x) for x in sorted(selected)) + ' → ' + label(target))
        st.write(f'Support: {joint/len(matrix):.2%} · Confidence: {conf:.2%} · Lift: {conf/base:.2f}')
        st.write('Dataset: MIMIC-IV Clinical Database Demo 2.2 (3-character ICD-9/ICD-10 disease groups)')
        st.download_button(
            'Download results for this selection',
            rows[['antecedent', 'consequent', 'antecedent_count', 'joint_count', 'support', 'confidence', 'baseline', 'lift']].to_csv(index=False).encode('utf-8-sig'),
            'app3_mimic_iv_results.csv',
            'text/csv',
        )
