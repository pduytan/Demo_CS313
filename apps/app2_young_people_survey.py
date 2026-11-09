"""Application 2 — Personality & Behavior Pattern Explorer (Young People Survey)."""
from pathlib import Path
from html import escape
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from apps.app2_dataset import prepare
from apps.app2_trait_labels import TRAITS
from shared.demo_helpers import allowed_extensions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'app2_young_people_survey'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'app2_young_people_survey'

NAMES = {
    key: (key, key) for key in [
        *TRAITS, 'Health', 'Only child', 'Gender', 'Music', 'Movies',
        'Personality', 'Biology', 'Chemistry', 'Gardening', 'Physics',
        'Left - right handed', 'Village - town', 'House - block of flats',
        'Internet usage', 'Age', 'Height', 'Weight', 'Number of siblings',
    ]
}
EMOTIONS = {'Getting angry', 'Mood swings', 'Loneliness', 'Life struggles', 'Happiness in life', 'Energy levels', 'Self-criticism'}


@st.cache_data
def load():
    raw, matrix, _ = prepare('exact')
    rules = pd.read_csv(RESULTS_DIR / 'association_rules.csv')
    rules['antecedents'] = rules.antecedent.map(lambda x: frozenset(x.split(' + ')))
    questions = pd.read_csv(DATA_DIR / 'columns.csv')
    return raw, matrix, rules, dict(zip(questions.iloc[:, 1], questions.iloc[:, 0]))


def label(item):
    return item


def describe(item, questions):
    col, value = item.rsplit(' = ', 1)
    content = str(questions.get(col, col)).rstrip('.')
    if value in ['1', '2', '3', '4', '5'] and col not in ['Number of siblings', 'Age', 'Height', 'Weight']:
        return f'select response {value}/5 for “{content}” ({col})'
    return f'report {col} = {value}'


def restore_selection(valid, previous):
    proposed = st.session_state.get('scenario_items', [])
    if proposed and frozenset(proposed) not in valid:
        st.session_state.scenario_items = list(previous)
        st.session_state._saved_scenario_items = list(previous)
        st.session_state.scenario_notice = 'This combination has no results under the current filters. Your previous selection was restored; clear all items to start again.'
    else:
        st.session_state._saved_scenario_items = list(proposed)


def change_topic():
    st.session_state.scenario_items = []
    st.session_state._saved_scenario_items = []


def rule_diagram(selected, target, n, joint):
    chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in sorted(selected))
    return (f'<div class="rule-flow"><div class="rule-node"><div class="rule-kicker">INPUT · ALL ITEMS PRESENT</div>'
            f'{chips}<div class="rule-note">{n} respondents match all input items</div></div>'
            '<div class="rule-arrow" aria-label="associated with">→</div>'
            f'<div class="rule-node rule-target"><div class="rule-kicker">ASSOCIATED RESPONSE</div>'
            f'<span class="rule-chip">{escape(target)}</span>'
            f'<div class="rule-note">{joint}/{n} respondents also match this item · {joint/n:.1%}</div></div></div>'
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


def distribution_note(matrix, matched, target):
    col, value = target.rsplit(' = ', 1)
    items = [x for x in matrix if x.rsplit(' = ', 1)[0] == col]
    overall = matrix[items].mean()
    group = matrix.loc[matched, items].mean()
    mode = group.idxmax().rsplit(' = ', 1)[1]
    changes = group - overall
    lines = [f'Within the selected group, response “{mode}” is the most common ({group.max():.1%}).']
    up, down = changes.idxmax(), changes.idxmin()
    if changes[up] > 0.00005:
        lines.append(f'Compared with the full sample, response “{up.rsplit(" = ", 1)[1]}” shows the largest increase '
                     f'({100*changes[up]:.1f} percentage points).')
    if changes[down] < -0.00005:
        lines.append(f'Response “{down.rsplit(" = ", 1)[1]}” shows the largest decrease '
                     f'({-100*changes[down]:.1f} percentage points).')
    lines.append(f'The current rule concerns response “{value}”; this distribution describes survey responses, not better or worse ability or personality.')
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
    used = {x.rsplit(' = ', 1)[0] for x in selected}
    rows = []
    for item in matrix:
        if item.rsplit(' = ', 1)[0] in used or not joint[item] or not base[item]:
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
                baseline=base[item],
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
    if len(selected) >= 4:
        return valid
    mask = matrix[list(selected)].all(axis=1) if selected else pd.Series(True, index=matrix.index)
    values = matrix.loc[mask].to_numpy(dtype=np.float64)
    counts = values.sum(axis=0)
    joints = values.T @ values
    base = matrix.mean().to_numpy()
    conf = np.divide(joints, counts[:, None], out=np.zeros_like(joints), where=counts[:, None] > 0)
    lifts = np.divide(conf, base[None, :], out=np.zeros_like(conf), where=base[None, :] > 0)
    names = np.array([x.rsplit(' = ', 1)[0] for x in matrix])
    used = {x.rsplit(' = ', 1)[0] for x in selected}
    targets = ~np.isin(names, list(used))
    ok = (joints > 0) & (joints / len(matrix) >= support - 1e-12) & (conf >= confidence - 1e-12) & (lifts > lift)
    ok &= names[:, None] != names[None, :]
    ok &= targets[None, :]
    for i, item in enumerate(matrix):
        if names[i] not in used and ok[i].any():
            valid.add(selected | {item})
    return valid


def main():
    _, matrix, rules, questions = load()
    st.markdown('''<div class="hero"><div class="eyebrow">YOUNG PEOPLE SURVEY</div>
    <h1>Discover Patterns in Personality &amp; Behavior</h1>
    <p>Explore how survey responses co-occur across personality, emotions, habits and interests.</p>
    <span class="badge">674 respondents &nbsp; · &nbsp; 150 features</span></div>''', unsafe_allow_html=True)
    st.subheader('01 · Choose your starting features')
    with st.sidebar:
        st.subheader('Result filters')
        support = st.slider('Support', 0.0, 1.0, .10, .01, key='scenario_support')
        confidence = st.slider('Confidence', 0.0, 1.0, .30, .01, key='scenario_confidence')
        lift = st.slider('Lift', 0.0, 5.0, 1.20, .05, key='scenario_lift')
    filtered = rules[rules.support.ge(support - 1e-12) & rules.confidence.ge(confidence - 1e-12) & rules.lift.gt(lift)]
    live_query = support < .10 or confidence < .30 or lift < 1.20
    prioritize_emotion = True
    valid = set(filtered.antecedents)
    if st.session_state.get('personality_navigation_version') != 1:
        st.session_state._saved_scenario_items = ['Keeping promises = 5']
        st.session_state.personality_navigation_version = 1
    if 'scenario_items' not in st.session_state:
        st.session_state.scenario_items = list(st.session_state.get('_saved_scenario_items', ['Keeping promises = 5']))
    if 'input_topic' not in st.session_state and '_saved_input_topic' in st.session_state:
        st.session_state.input_topic = st.session_state._saved_input_topic
    topic = st.selectbox(
        'Browse features by topic',
        ['Personality & habits', 'Emotions & states', 'All features'],
        key='input_topic',
        on_change=change_topic,
    )
    st.session_state._saved_input_topic = topic
    selected = frozenset(st.session_state.get('scenario_items', []))
    if live_query:
        valid = query_navigation(matrix, selected, support, confidence, lift)
    if selected and selected not in valid:
        st.session_state.scenario_items = []
        st.session_state._saved_scenario_items = []
        selected = frozenset()
        if live_query:
            valid = query_navigation(matrix, selected, support, confidence, lift)
        st.info('Your previous selection no longer meets the filters. Start again with the available items below.')
    ordered = sorted(
        matrix.columns,
        key=lambda x: (
            prioritize_emotion and x.rsplit(' = ', 1)[0] not in TRAITS,
            x.rsplit(' = ', 1)[0] not in NAMES,
            label(x),
        ),
    )
    options = allowed_extensions(valid, selected, ordered)
    topic_keys = (
        set(TRAITS) - EMOTIONS if topic == 'Personality & habits'
        else EMOTIONS if topic == 'Emotions & states'
        else set(matrix.columns.str.rsplit(' = ', n=1).str[0])
    )
    options = [x for x in options if x in selected or x.rsplit(' = ', 1)[0] in topic_keys]
    st.multiselect(
        'Select 1–4 feature–response items',
        options,
        format_func=label,
        max_selections=4,
        key='scenario_items',
        on_change=restore_selection,
        args=(valid, tuple(st.session_state.get('scenario_items', []))),
        disabled=not options,
    )
    if st.session_state.get('scenario_notice'):
        st.info(st.session_state.pop('scenario_notice'))
    current_items = st.session_state.get('scenario_items', [])
    st.session_state._saved_scenario_items = list(current_items)
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
    candidates['emotion'] = candidates.consequent.map(lambda x: x.rsplit(' = ', 1)[0] in EMOTIONS)
    candidates['personality'] = candidates.consequent.map(lambda x: x.rsplit(' = ', 1)[0] in TRAITS)
    if prioritize_emotion:
        candidates = candidates.sort_values(['personality', 'confidence', 'lift'], ascending=False)
    st.divider()
    st.subheader('02 · Explore associated responses')
    st.caption(f'The selected group contains {n} of {len(matrix)} respondents.')
    if candidates.empty:
        st.info('No results meet the current filters.')
        return
    personality_count = int(candidates.personality.sum())
    if prioritize_emotion and not personality_count:
        st.info('This combination has no personality or behavior results under the current filters. Other associated responses are shown below.')
    rows = candidates.copy()
    rows['Response'] = rows.consequent.map(label)
    st.markdown(result_table(rows), unsafe_allow_html=True)
    result_options = list(candidates.consequent)
    initial_result = next(
        (item for item in result_options if 'Only child = yes' in selected and item.startswith('Health = ')),
        result_options[0],
    )
    if prioritize_emotion and personality_count:
        initial_result = result_options[0]
    if st.session_state.get('scenario_result') not in result_options:
        st.session_state.scenario_result = initial_result
    chosen = st.selectbox(
        'Choose a response to examine',
        result_options,
        index=result_options.index(st.session_state.get('scenario_result', initial_result)),
        format_func=label,
        key='scenario_result',
    )
    row = candidates[candidates.consequent.eq(chosen)].iloc[0]
    target = row.consequent
    joint = int(matrix[list(selected | {target})].all(axis=1).sum())
    base = float(matrix[target].mean())
    conf = joint / n
    condition = ' and '.join(describe(x, questions) for x in sorted(selected))
    outcome = describe(target, questions)
    st.divider()
    st.subheader('03 · Understand the selected association')
    st.markdown(rule_diagram(selected, target, n, joint), unsafe_allow_html=True)
    st.markdown('**If a respondent were to ' + condition + '…**')
    st.markdown('**Associated response in this sample:** ' + outcome + '.')
    st.write(f'Of the **{n} respondents** matching all selected items, **{joint} ({conf:.1%})** also {outcome}.')
    direction = 'higher' if conf >= base else 'lower'
    st.write(f'In the full sample, this response occurs in **{base:.1%}** of respondents. The selected group is '
             f'**{100*abs(conf-base):.1f} percentage points {direction}**, or **{conf/base:.2f} times** the baseline rate.')
    if target.startswith('Health = ') or any(x.startswith('Health = ') for x in selected):
        st.info('Health measures agreement with “I worry about my health.” It reflects reported worry, not whether someone is healthy or unhealthy.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Support', f'{joint/len(matrix):.2%}',
              help='Respondents matching both the input and outcome, divided by the full sample.')
    c1.caption(f'{joint}/{len(matrix)} respondents in the full sample')
    c2.metric('Confidence', f'{conf:.2%}',
              help='Respondents matching both the input and outcome, divided by those matching the input.')
    c2.caption(f'{joint}/{n} respondents in the selected group')
    c3.metric('Lift', f'{conf/base:.2f}×',
              help='Confidence divided by the outcome frequency in the full sample.')
    c3.caption(f'{conf:.2%} / {base:.2%}')
    target_column = target.rsplit(' = ', 1)[0]
    chart_context = (tuple(sorted(selected)), target, float(support), float(confidence), float(lift))
    if st.session_state.get('chart_context') != chart_context:
        st.session_state.chart_context = chart_context
        st.session_state.show_analysis_charts = False
    if st.button('Hide analysis charts' if st.session_state.get('show_analysis_charts') else 'Show analysis charts',
                 key='show_charts_button', icon=':material/monitoring:', type='primary'):
        st.session_state.show_analysis_charts = not st.session_state.get('show_analysis_charts', False)
        st.rerun()
    if st.session_state.get('show_analysis_charts'):
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('**How common is this response?**')
            fig = px.bar(
                pd.DataFrame({'Group': ['Full sample', 'Selected group'], 'Proportion': [base, conf]}),
                x='Group', y='Proportion', color='Group',
                color_discrete_sequence=['#b7a4b1', '#3a0f34'], text_auto='.1%',
            )
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=410, margin=dict(t=30, b=25),
                yaxis=dict(range=[0, 1], tickformat='.0%'),
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(f'Response “{target.rsplit(" = ", 1)[1]}” occurs in {conf:.1%} of the selected group, '
                       f'versus {base:.1%} of the full sample: {100*abs(conf-base):.1f} percentage points {direction} '
                       f'(lift {conf/base:.2f}). The full sample includes the selected group.')
        with right:
            st.markdown('**Within the selected group**')
            fig = go.Figure(go.Pie(
                labels=['Exact selected response', 'Other responses'],
                values=[joint, n - joint], hole=.65,
                marker_colors=['#3a0f34', '#d6cdd2'], textinfo='value+percent',
            ))
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=410, margin=dict(t=20, b=20), legend=dict(orientation='h'),
                annotations=[dict(text=f'{joint}/{n}', x=.5, y=.5, font_size=18, showarrow=False)],
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(f'Of {n} respondents matching all input items, {joint} selected response '
                       f'“{target.rsplit(" = ", 1)[1]}”; {n-joint} ({1-conf:.1%}) selected a different response '
                       'to the same question. A different response does not imply the opposite trait.')
        target_column = target.rsplit(' = ', 1)[0]
        same_question = [x for x in matrix if x.rsplit(' = ', 1)[0] == target_column]
        distributions = []
        for item in same_question:
            value = item.rsplit(' = ', 1)[1]
            distributions.extend([
                {'Response value': value, 'Group': 'Full sample', 'Proportion': float(matrix[item].mean())},
                {'Response value': value, 'Group': 'Selected group', 'Proportion': float(matrix.loc[matched, item].mean())},
            ])
        st.markdown('**How does the full response distribution compare?**')
        st.caption(str(questions.get(target_column, target_column)))
        fig = px.bar(
            pd.DataFrame(distributions), x='Response value', y='Proportion', color='Group', barmode='group',
            color_discrete_map={'Full sample': '#b7a4b1', 'Selected group': '#3a0f34'}, text_auto='.1%',
        )
        fig.update_layout(
            template='plotly_white', font=dict(size=18, color='#523c4d'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=430, yaxis=dict(tickformat='.0%', range=[0, 1]), margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        chart_note(distribution_note(matrix, matched, target))
    st.caption('A different response does not imply the opposite trait. These are survey associations, not causal effects or guaranteed predictions for individuals.')
    with st.expander('View itemset and detailed figures'):
        st.write('**Itemset:** ' + ' + '.join(label(x) for x in sorted(selected | {target})))
        st.write('**Rule:** ' + ' + '.join(label(x) for x in sorted(selected)) + ' → ' + label(target))
        st.write(f'Support: {joint/len(matrix):.2%} · Confidence: {conf:.2%} · Lift: {conf/base:.2f}')
        st.write('Original question: ' + str(questions.get(target_column, target_column)))
        st.download_button(
            'Download results for this selection',
            rows[['antecedent', 'consequent', 'antecedent_count', 'joint_count', 'support', 'confidence', 'baseline', 'lift']].to_csv(index=False).encode('utf-8-sig'),
            'app2_young_people_survey_results.csv',
            'text/csv',
        )
