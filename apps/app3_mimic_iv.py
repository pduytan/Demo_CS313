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
        return 'Tim mạch, Chuyển hóa & Nội tiết'
    if any(code.startswith(p) for p in RENAL_RESP_HEME_PREFIXES):
        return 'Thận, Hô hấp & Huyết học'
    return 'Tiêu hóa, Thần kinh & Nhóm khác'


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
        formatted.append(f'nhóm bệnh “{title}” ({code})')
    return 'có ghi nhận ' + ' và '.join(formatted)


def restore_selection(valid, previous):
    proposed = st.session_state.get('app3_items', [])
    if proposed and frozenset(proposed) not in valid:
        st.session_state.app3_items = list(previous)
        st.session_state._saved_app3_items = list(previous)
        st.session_state.app3_notice = 'Tổ hợp này không có kết quả thỏa mãn bộ lọc hiện tại. Đã khôi phục lựa chọn trước đó; hãy xóa các mục đã chọn để bắt đầu lại.'
    else:
        st.session_state._saved_app3_items = list(proposed)


def change_topic():
    st.session_state.app3_items = []
    st.session_state._saved_app3_items = []


def rule_diagram(selected, target, n, joint):
    chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in sorted(selected))
    target_chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in target.split(' + '))
    return (f'<div class="rule-flow"><div class="rule-node"><div class="rule-kicker">ĐẦU VÀO · CHẨN ĐOÁN HIỆN CÓ</div>'
            f'{chips}<div class="rule-note">{n} bệnh nhân có đủ các chẩn đoán đầu vào</div></div>'
            '<div class="rule-arrow" aria-label="associated with">→</div>'
            f'<div class="rule-node rule-target"><div class="rule-kicker">CHẨN ĐOÁN ĐỒNG MẮC</div>'
            f'{target_chips}'
            f'<div class="rule-note">{joint}/{n} bệnh nhân cũng có chẩn đoán này · {joint/n:.1%}</div></div></div>'
            '<div class="rule-footnote">Mũi tên thể hiện chiều của luật kết hợp (association rule), không mang ý nghĩa nhân quả.</div>')


def chart_note(text):
    st.markdown(f'<div class="chart-note"><strong>Nhận xét</strong><p>{escape(text)}</p></div>', unsafe_allow_html=True)


def result_table(rows):
    body = ''.join(
        f'<tr><td>{escape(str(row.consequent))}</td><td>{int(row.joint_count):,}</td>'
        f'<td>{row.confidence:.2%}</td><td>{row.lift:.2f}×</td></tr>'
        for row in rows.head(20).itertuples())
    return ('<div class="results-table-wrap" role="region" aria-label="Associated responses" tabindex="0">'
            '<table class="results-table"><thead><tr><th scope="col">Chẩn đoán đồng mắc</th>'
            '<th scope="col">Số bệnh nhân chung</th><th scope="col">Confidence</th>'
            '<th scope="col">Lift</th></tr></thead><tbody>' + body + '</tbody></table></div>')


def distribution_note(matrix, matched, target, peer_items):
    overall = matrix[peer_items].mean()
    group = matrix.loc[matched, peer_items].mean()
    top_item = group.idxmax()
    changes = group - overall
    up = changes.idxmax()
    lines = [
        f'Trong nhóm bệnh nhân đã chọn, “{top_item}” có tỷ lệ hiện diện cao nhất ({group.max():.1%}).',
        f'So với toàn bộ {len(matrix)} bệnh nhân, “{up}” có mức tăng lớn nhất (+{100*changes[up]:.1f}%).',
        f'Luật hiện tại xét “{target}”; các liên kết này phản ánh tình trạng bệnh đồng mắc (comorbidity) trong mẫu bệnh viện, không khẳng định bệnh này trực tiếp gây ra bệnh kia.'
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
    st.markdown(f'''<div class="hero"><div class="eyebrow">CƠ SỞ DỮ LIỆU LÂM SÀNG MIMIC-IV DEMO 2.2</div>
    <h1>Khám phá Quy luật Bệnh đồng mắc</h1>
    <p>Khám phá cách các nhóm bệnh ICD-9 và ICD-10 đồng xuất hiện trong hồ sơ bệnh án nội trú.</p>
    <span class="badge">{len(matrix)} bệnh nhân &nbsp; · &nbsp; {total_icd3} nhóm bệnh ICD-3</span></div>''', unsafe_allow_html=True)
    st.subheader('01 · Chọn các chẩn đoán khởi đầu')
    with st.sidebar:
        st.subheader('Bộ lọc kết quả')
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
    topics = ['Tim mạch, Chuyển hóa & Nội tiết', 'Thận, Hô hấp & Huyết học', 'Tiêu hóa, Thần kinh & Nhóm khác', 'Tất cả chẩn đoán']
    if st.session_state.get('_saved_app3_topic') in topics and 'app3_topic' not in st.session_state:
        st.session_state.app3_topic = st.session_state._saved_app3_topic
    elif st.session_state.get('app3_topic') not in topics:
        st.session_state.app3_topic = topics[0]
    topic = st.selectbox(
        'Lọc nhóm bệnh theo hệ cơ quan lâm sàng',
        topics,
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
        st.info('Lựa chọn trước đó không còn thỏa mãn bộ lọc hiện tại. Hãy chọn lại từ danh sách bên dưới.')
    ordered = sorted(matrix.columns, key=lambda x: (classify_diagnosis(x) == 'Tiêu hóa, Thần kinh & Nhóm khác', label(x)))
    options = allowed_extensions(valid, selected, ordered)
    if topic != 'Tất cả chẩn đoán':
        options = [x for x in options if x in selected or classify_diagnosis(x) == topic]
    st.multiselect(
        'Chọn từ 1–3 nhóm bệnh chẩn đoán đầu vào',
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
        st.info('Hãy chọn một chẩn đoán để bắt đầu.' if valid else 'Không có luật nào thỏa mãn bộ lọc hiện tại. Hãy giảm ngưỡng lọc ở thanh bên trái.')
        return
    matched = matrix[list(selected)].all(axis=1)
    n = int(matched.sum())
    candidates = filtered[filtered.antecedents.eq(selected)].sort_values(['confidence', 'lift'], ascending=False)
    if live_query:
        candidates = query_rules(matrix, selected, support, confidence, lift).sort_values(['confidence', 'lift'], ascending=False)
    candidates = candidates.copy()
    st.divider()
    st.subheader('02 · Khám phá các chẩn đoán đồng mắc')
    st.caption(f'Nhóm đang chọn gồm {n} trên tổng số {len(matrix)} bệnh nhân.')
    if candidates.empty:
        ext_rules = filtered[filtered.antecedents.map(lambda a: selected.issubset(a) and len(a) > len(selected))]
        direct_fallback = rules[rules.antecedents.eq(selected)].sort_values(['confidence', 'lift'], ascending=False)
        if direct_fallback.empty:
            direct_fallback = query_rules(matrix, selected, 0.05, 0.20, 1.00).sort_values(['confidence', 'lift'], ascending=False)
        if not ext_rules.empty:
            companions = sorted({x for a in ext_rules.antecedents for x in (a - selected)})
            st.info(
                f'Chẩn đoán này tham gia vào các luật kết hợp đa bệnh lý ở ngưỡng hiện tại (Confidence ≥ {confidence:.0%}). '
                f'Hãy chọn thêm chẩn đoán đi kèm ở Bước 01 ({", ".join(companions[:3])}) để xem luật đa mục, hoặc xem các liên kết trực tiếp bên dưới.'
            )
        elif not direct_fallback.empty:
            st.info(
                f'Khi đứng độc lập ở vế trái, chẩn đoán này có Confidence tối đa là {direct_fallback.confidence.max():.1%} '
                f'(thấp hơn ngưỡng lọc {confidence:.0%} hiện tại). Các chẩn đoán đồng mắc trực tiếp được hiển thị bên dưới.'
            )
        candidates = direct_fallback.copy()
    if candidates.empty:
        st.info('Không có kết quả nào thỏa mãn bộ lọc hiện tại.')
        return
    rows = candidates.copy()
    rows['Response'] = rows.consequent.map(label)
    st.markdown(result_table(rows), unsafe_allow_html=True)
    result_options = list(candidates.consequent)
    initial_result = result_options[0]
    if st.session_state.get('app3_result') not in result_options:
        st.session_state.app3_result = initial_result
    chosen = st.selectbox(
        'Chọn một chẩn đoán đồng mắc để phân tích chi tiết',
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
    condition = ' và '.join(describe(x) for x in sorted(selected))
    outcome = describe(target)
    st.divider()
    st.subheader('03 · Phân tích chi tiết luật kết hợp đã chọn')
    st.markdown(rule_diagram(selected, target, n, joint), unsafe_allow_html=True)
    st.markdown('**Nếu hồ sơ bệnh án của một bệnh nhân ' + condition + '…**')
    st.markdown('**Chẩn đoán đồng mắc trong mẫu dữ liệu:** ' + outcome + '.')
    st.write(f'Trong số **{n} bệnh nhân** có đủ các chẩn đoán đầu vào, có **{joint} bệnh nhân ({conf:.1%})** cũng {outcome}.')
    direction = 'cao hơn' if conf >= base else 'thấp hơn'
    st.write(f'Trên toàn bộ mẫu bệnh nhân, chẩn đoán này xuất hiện ở **{base:.1%}** bệnh nhân. Nhóm đã chọn có tỷ lệ '
             f'**{direction} {100*abs(conf-base):.1f}%**, tương đương **{conf/base:.2f} lần** mức nền (baseline).')
    st.info('Các chẩn đoán được gom theo mã ICD-9/ICD-10 3 ký tự (đã loại bỏ các mã ngoại nhân và triệu chứng V/E/R/W/X/Y/Z) và đặt tên theo mã ICD chi tiết phổ biến nhất trong từng nhóm.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Support', f'{joint/len(matrix):.2%}',
              help='Số bệnh nhân có cả chẩn đoán đầu vào và chẩn đoán kết quả chia cho tổng số bệnh nhân.')
    c1.caption(f'{joint}/{len(matrix)} bệnh nhân trong toàn bộ mẫu')
    c2.metric('Confidence', f'{conf:.2%}',
              help='Số bệnh nhân có cả chẩn đoán đầu vào và kết quả chia cho số bệnh nhân có chẩn đoán đầu vào.')
    c2.caption(f'{joint}/{n} bệnh nhân trong nhóm đã chọn')
    c3.metric('Lift', f'{conf/base:.2f}×',
              help='Confidence chia cho tần suất xuất hiện nền của chẩn đoán kết quả trong toàn mẫu.')
    c3.caption(f'{conf:.2%} / {base:.2%}')

    chart_context = ('app3', tuple(sorted(selected)), target, float(support), float(confidence), float(lift))
    if st.session_state.get('app3_chart_context') != chart_context:
        st.session_state.app3_chart_context = chart_context
        st.session_state.app3_show_charts = False
    if st.button('Ẩn biểu đồ phân tích' if st.session_state.get('app3_show_charts') else 'Hiện biểu đồ phân tích',
                 key='show_charts_button', icon=':material/monitoring:', type='primary'):
        st.session_state.app3_show_charts = not st.session_state.get('app3_show_charts', False)
        st.rerun()
    if st.session_state.get('app3_show_charts'):
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('**Mức độ phổ biến của chẩn đoán này như thế nào?**')
            fig = px.bar(
                pd.DataFrame({'Nhóm': ['Toàn bộ mẫu', 'Nhóm đã chọn'], 'Tỷ lệ': [base, conf]}),
                x='Nhóm', y='Tỷ lệ', color='Nhóm',
                color_discrete_sequence=['#b7a4b1', '#3a0f34'], text_auto='.1%'
            )
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=410, margin=dict(t=30, b=25),
                yaxis=dict(range=[0, 1], tickformat='.0%')
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(f'Chẩn đoán “{target}” xuất hiện ở {conf:.1%} nhóm bệnh nhân đã chọn, '
                       f'so với {base:.1%} của toàn bộ mẫu: {direction} {100*abs(conf-base):.1f}% '
                       f'(Lift {conf/base:.2f}). Toàn bộ mẫu đã bao gồm nhóm được chọn.')
        with right:
            st.markdown('**Tỷ lệ bên trong nhóm bệnh nhân đã chọn**')
            fig = go.Figure(go.Pie(
                labels=['Có chẩn đoán đích', 'Không có chẩn đoán đích'],
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
            chart_note(f'Trong số {n} bệnh nhân có đủ chẩn đoán đầu vào, có {joint} bệnh nhân cũng được ghi nhận '
                       f'“{target}”; {n-joint} bệnh nhân ({1-conf:.1%}) không ghi nhận chẩn đoán này.')
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
                    {'Nhóm ICD-3': short_code, 'Nhóm': 'Toàn bộ mẫu', 'Tỷ lệ': float(matrix[item].mean())},
                    {'Nhóm ICD-3': short_code, 'Nhóm': 'Nhóm đã chọn', 'Tỷ lệ': float(matrix.loc[matched, item].mean())},
                ])
            st.markdown('**So sánh hồ sơ các nhóm bệnh đồng mắc hàng đầu**')
            st.caption('So sánh tỷ lệ lưu hành nền (toàn mẫu) và tỷ lệ có điều kiện (nhóm đã chọn) qua các nhóm bệnh ICD-3 liên kết mạnh nhất.')
            fig = px.bar(
                pd.DataFrame(distributions), x='Nhóm ICD-3', y='Tỷ lệ', color='Nhóm', barmode='group',
                color_discrete_map={'Toàn bộ mẫu': '#b7a4b1', 'Nhóm đã chọn': '#3a0f34'}, text_auto='.1%'
            )
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=430, yaxis=dict(tickformat='.0%', range=[0, 1]), margin=dict(t=30, b=20)
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(distribution_note(matrix, matched, target, peer_items))
    st.caption('Sự đồng xuất hiện của các mã chẩn đoán xuất viện phản ánh mô hình bệnh đồng mắc trong mẫu nghiên cứu, không khẳng định tiến triển nhân quả hay chẩn đoán chắc chắn.')
    with st.expander('Xem tập mục (Itemset) và số liệu chi tiết'):
        st.write('**Itemset:** ' + ' + '.join(label(x) for x in sorted(selected | target_items)))
        st.write('**Luật kết hợp:** ' + ' + '.join(label(x) for x in sorted(selected)) + ' → ' + label(target))
        st.write(f'Support: {joint/len(matrix):.2%} · Confidence: {conf:.2%} · Lift: {conf/base:.2f}')
        st.write('Bộ dữ liệu: MIMIC-IV Clinical Database Demo 2.2 (Nhóm bệnh ICD-9/ICD-10 3 ký tự)')
        st.download_button(
            'Tải kết quả cho lựa chọn này (CSV)',
            rows[['antecedent', 'consequent', 'antecedent_count', 'joint_count', 'support', 'confidence', 'baseline', 'lift']].to_csv(index=False).encode('utf-8-sig'),
            'app3_mimic_iv_results.csv',
            'text/csv',
        )
