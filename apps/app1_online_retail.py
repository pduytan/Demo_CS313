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
        return 'Túi xách & Lưu trữ'
    if any(k in upper for k in KITCHEN_KW):
        return 'Tách trà, Nhà bếp & Bàn ăn'
    if any(k in upper for k in DECOR_KW):
        return 'Trang trí nhà cửa & Quà tặng'
    return 'Sản phẩm khác'


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
        formatted.append(f'sản phẩm “{name}” ({code})')
    return 'có mua ' + ' và '.join(formatted)


def restore_selection(valid, previous):
    proposed = st.session_state.get('app1_items', [])
    if proposed and frozenset(proposed) not in valid:
        st.session_state.app1_items = list(previous)
        st.session_state._saved_app1_items = list(previous)
        st.session_state.app1_notice = 'Tổ hợp này không có kết quả thỏa mãn bộ lọc hiện tại. Đã khôi phục lựa chọn trước đó; hãy xóa các mục đã chọn để bắt đầu lại.'
    else:
        st.session_state._saved_app1_items = list(proposed)


def change_topic():
    st.session_state.app1_items = []
    st.session_state._saved_app1_items = []


def rule_diagram(selected, target, n, joint):
    chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in sorted(selected))
    target_chips = ''.join(f'<span class="rule-chip">{escape(x)}</span>' for x in target.split(' + '))
    return (f'<div class="rule-flow"><div class="rule-node"><div class="rule-kicker">ĐẦU VÀO · CÁC SẢN PHẨM TRONG GIỎ</div>'
            f'{chips}<div class="rule-note">{n:,} hóa đơn có tất cả sản phẩm đầu vào</div></div>'
            '<div class="rule-arrow" aria-label="associated with">→</div>'
            f'<div class="rule-node rule-target"><div class="rule-kicker">SẢN PHẨM MUA KÈM</div>'
            f'{target_chips}'
            f'<div class="rule-note">{joint:,}/{n:,} hóa đơn cũng có sản phẩm này · {joint/n:.1%}</div></div></div>'
            '<div class="rule-footnote">Mũi tên thể hiện chiều của luật kết hợp (association rule), không mang ý nghĩa nhân quả.</div>')


def chart_note(text):
    st.markdown(f'<div class="chart-note"><strong>Nhận xét</strong><p>{escape(text)}</p></div>', unsafe_allow_html=True)


def result_table(rows):
    body = ''.join(
        f'<tr><td>{escape(str(row.consequent))}</td><td>{int(row.joint_count):,}</td>'
        f'<td>{row.confidence:.2%}</td><td>{row.lift:.2f}×</td></tr>'
        for row in rows.head(20).itertuples())
    return ('<div class="results-table-wrap" role="region" aria-label="Associated responses" tabindex="0">'
            '<table class="results-table"><thead><tr><th scope="col">Sản phẩm đi kèm</th>'
            '<th scope="col">Số hóa đơn chung</th><th scope="col">Confidence</th>'
            '<th scope="col">Lift</th></tr></thead><tbody>' + body + '</tbody></table></div>')


def distribution_note(matrix, matched, target, peer_items):
    overall = matrix[peer_items].mean()
    group = matrix.loc[matched, peer_items].mean()
    top_item = group.idxmax()
    changes = group - overall
    up = changes.idxmax()
    lines = [
        f'Trong nhóm hóa đơn đã chọn, “{top_item}” có tỷ lệ mua kèm cao nhất ({group.max():.1%}).',
        f'So với toàn bộ {len(matrix):,} hóa đơn UK, “{up}” có mức tăng lớn nhất (+{100*changes[up]:.1f}%).',
        f'Luật đang xét tập trung vào “{target}”; các tỷ lệ đồng xuất hiện này phản ánh giỏ hàng thực tế, không hàm ý quan hệ nhân quả.'
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
    st.markdown(f'''<div class="hero"><div class="eyebrow">UCI ONLINE RETAIL · UK</div>
    <h1>Khám phá Quy luật Giỏ hàng</h1>
    <p>Khám phá cách các sản phẩm bán lẻ xuất hiện cùng nhau trên hóa đơn khách hàng để gợi ý combo và bán kèm.</p>
    <span class="badge">{len(matrix):,} hóa đơn UK hợp lệ &nbsp; · &nbsp; {total_products:,} sản phẩm</span></div>''', unsafe_allow_html=True)
    st.subheader('01 · Chọn sản phẩm khởi đầu trong giỏ hàng')
    with st.sidebar:
        st.subheader('Bộ lọc kết quả')
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
    topics = ['Túi xách & Lưu trữ', 'Tách trà, Nhà bếp & Bàn ăn', 'Trang trí nhà cửa & Quà tặng', 'Tất cả sản phẩm']
    if st.session_state.get('_saved_app1_topic') in topics and 'app1_topic' not in st.session_state:
        st.session_state.app1_topic = st.session_state._saved_app1_topic
    elif st.session_state.get('app1_topic') not in topics:
        st.session_state.app1_topic = topics[0]
    topic = st.selectbox(
        'Lọc danh mục sản phẩm theo chủ đề',
        topics,
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
        st.info('Lựa chọn trước đó không còn thỏa mãn bộ lọc hiện tại. Hãy chọn lại từ danh sách bên dưới.')
    ordered = sorted(matrix.columns, key=lambda x: (classify_product(x) == 'Sản phẩm khác', label(x)))
    options = allowed_extensions(valid, selected, ordered)
    if topic != 'Tất cả sản phẩm':
        options = [x for x in options if x in selected or classify_product(x) == topic]
    st.multiselect(
        'Chọn từ 1–3 sản phẩm đầu vào',
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
        st.info('Hãy chọn một sản phẩm để bắt đầu.' if valid else 'Không có luật nào thỏa mãn bộ lọc hiện tại. Hãy giảm ngưỡng lọc ở thanh bên trái.')
        return
    matched = matrix[list(selected)].all(axis=1)
    n = int(matched.sum())
    candidates = filtered[filtered.antecedents.eq(selected)].sort_values(['confidence', 'lift'], ascending=False)
    if live_query:
        candidates = query_rules(matrix, selected, support, confidence, lift).sort_values(['confidence', 'lift'], ascending=False)
    candidates = candidates.copy()
    st.divider()
    st.subheader('02 · Khám phá các sản phẩm mua kèm')
    st.caption(f'Nhóm đang chọn gồm {n:,} trên tổng số {len(matrix):,} hóa đơn UK hợp lệ.')
    if candidates.empty:
        ext_rules = filtered[filtered.antecedents.map(lambda a: selected.issubset(a) and len(a) > len(selected))]
        direct_fallback = rules[rules.antecedents.eq(selected)].sort_values(['confidence', 'lift'], ascending=False)
        if direct_fallback.empty:
            direct_fallback = query_rules(matrix, selected, 0.005, 0.10, 1.20).sort_values(['confidence', 'lift'], ascending=False)
        if not ext_rules.empty:
            companions = sorted({x for a in ext_rules.antecedents for x in (a - selected)})
            st.info(
                f'Sản phẩm này tham gia vào các luật kết hợp nhiều sản phẩm ở ngưỡng hiện tại (Confidence ≥ {confidence:.0%}). '
                f'Hãy chọn thêm sản phẩm đi cùng ở Bước 01 ({", ".join(companions[:3])}) để xem luật kết hợp đa mục, hoặc xem các liên kết trực tiếp bên dưới.'
            )
        elif not direct_fallback.empty:
            st.info(
                f'Khi đứng độc lập ở vế trái, sản phẩm phổ biến này có Confidence tối đa là {direct_fallback.confidence.max():.1%} '
                f'(thấp hơn ngưỡng lọc {confidence:.0%} hiện tại). Các sản phẩm mua kèm trực tiếp được hiển thị bên dưới.'
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
    if st.session_state.get('app1_result') not in result_options:
        st.session_state.app1_result = initial_result
    chosen = st.selectbox(
        'Chọn một sản phẩm mua kèm để phân tích chi tiết',
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
    condition = ' và '.join(describe(x) for x in sorted(selected))
    outcome = describe(target)
    st.divider()
    st.subheader('03 · Phân tích chi tiết luật kết hợp đã chọn')
    st.markdown(rule_diagram(selected, target, n, joint), unsafe_allow_html=True)
    st.markdown('**Nếu một hóa đơn ' + condition + '…**')
    st.markdown('**Sản phẩm mua kèm trong mẫu dữ liệu:** ' + outcome + '.')
    st.write(f'Trong số **{n:,} hóa đơn** có đủ các sản phẩm đầu vào, có **{joint:,} hóa đơn ({conf:.1%})** cũng {outcome}.')
    direction = 'cao hơn' if conf >= base else 'thấp hơn'
    st.write(f'Trên toàn bộ mẫu dữ liệu, sản phẩm này xuất hiện ở **{base:.2%}** tổng số hóa đơn. Tỷ lệ trong nhóm đã chọn '
             f'**{direction} {100*abs(conf-base):.1f}%**, tương đương **{conf/base:.2f} lần** mức nền (baseline).')
    st.info('Luật giỏ hàng phản ánh tần suất đồng xuất hiện trên các hóa đơn hợp lệ tại thị trường UK (đã loại hóa đơn hủy và mã phí vận chuyển), phục vụ gợi ý combo và bán chéo.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Support', f'{joint/len(matrix):.2%}',
              help='Số hóa đơn chứa cả vế trái và vế phải chia cho tổng số hóa đơn hợp lệ.')
    c1.caption(f'{joint:,}/{len(matrix):,} hóa đơn trong toàn bộ mẫu')
    c2.metric('Confidence', f'{conf:.2%}',
              help='Số hóa đơn chứa cả vế trái và vế phải chia cho số hóa đơn chứa vế trái.')
    c2.caption(f'{joint:,}/{n:,} hóa đơn trong nhóm đã chọn')
    c3.metric('Lift', f'{conf/base:.2f}×',
              help='Confidence chia cho tần suất xuất hiện nền của sản phẩm vế phải trong toàn mẫu.')
    c3.caption(f'{conf:.2%} / {base:.2%}')

    chart_context = ('app1', tuple(sorted(selected)), target, float(support), float(confidence), float(lift))
    if st.session_state.get('app1_chart_context') != chart_context:
        st.session_state.app1_chart_context = chart_context
        st.session_state.app1_show_charts = False
    if st.button('Ẩn biểu đồ phân tích' if st.session_state.get('app1_show_charts') else 'Hiện biểu đồ phân tích',
                 key='show_charts_button', icon=':material/monitoring:', type='primary'):
        st.session_state.app1_show_charts = not st.session_state.get('app1_show_charts', False)
        st.rerun()
    if st.session_state.get('app1_show_charts'):
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('**Mức độ phổ biến của sản phẩm này như thế nào?**')
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
            chart_note(f'Sản phẩm “{target}” xuất hiện trong {conf:.1%} nhóm hóa đơn đã chọn, '
                       f'so với {base:.2%} trên toàn bộ mẫu: {direction} {100*abs(conf-base):.1f}% '
                       f'(Lift {conf/base:.2f}). Toàn bộ mẫu đã bao gồm nhóm được chọn.')
        with right:
            st.markdown('**Tỷ lệ bên trong nhóm hóa đơn đã chọn**')
            fig = go.Figure(go.Pie(
                labels=['Có sản phẩm đích', 'Không có sản phẩm đích'],
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
            chart_note(f'Trong {n:,} hóa đơn có đủ sản phẩm đầu vào, có {joint:,} hóa đơn chứa '
                       f'“{target}”; {n-joint:,} hóa đơn ({1-conf:.1%}) không chứa sản phẩm này.')
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
                    {'Mã sản phẩm': short_code, 'Nhóm': 'Toàn bộ mẫu', 'Tỷ lệ': float(matrix[item].mean())},
                    {'Mã sản phẩm': short_code, 'Nhóm': 'Nhóm đã chọn', 'Tỷ lệ': float(matrix.loc[matched, item].mean())},
                ])
            st.markdown('**So sánh hồ sơ các sản phẩm mua kèm hàng đầu**')
            st.caption('So sánh tỷ lệ xuất hiện nền (toàn mẫu) và tỷ lệ có điều kiện (nhóm đã chọn) của các sản phẩm liên kết mạnh nhất.')
            fig = px.bar(
                pd.DataFrame(distributions), x='Mã sản phẩm', y='Tỷ lệ', color='Nhóm', barmode='group',
                color_discrete_map={'Toàn bộ mẫu': '#b7a4b1', 'Nhóm đã chọn': '#3a0f34'}, text_auto='.1%'
            )
            fig.update_layout(
                template='plotly_white', font=dict(size=18, color='#523c4d'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=430, yaxis=dict(tickformat='.0%', range=[0, 1]), margin=dict(t=30, b=20)
            )
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            chart_note(distribution_note(matrix, matched, target, peer_items))
    st.caption('Sự đồng xuất hiện trong lịch sử hóa đơn thể hiện xu hướng mua kèm, không đồng nghĩa với quan hệ nhân quả hay chắc chắn đúng cho mọi khách hàng.')
    with st.expander('Xem tập mục (Itemset) và số liệu chi tiết'):
        st.write('**Itemset:** ' + ' + '.join(label(x) for x in sorted(selected | target_items)))
        st.write('**Luật kết hợp:** ' + ' + '.join(label(x) for x in sorted(selected)) + ' → ' + label(target))
        st.write(f'Support: {joint/len(matrix):.2%} · Confidence: {conf:.2%} · Lift: {conf/base:.2f}')
        st.write('Bộ dữ liệu: UCI Online Retail (Các hóa đơn hợp lệ tại thị trường UK)')
        st.download_button(
            'Tải kết quả cho lựa chọn này (CSV)',
            rows[['antecedent', 'consequent', 'antecedent_count', 'joint_count', 'support', 'confidence', 'baseline', 'lift']].to_csv(index=False).encode('utf-8-sig'),
            'app1_online_retail_results.csv',
            'text/csv',
        )
