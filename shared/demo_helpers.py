"""Pure helpers for evidence-guided input and deterministic explanations."""


def allowed_extensions(valid_antecedents, selected, ordered_items):
    chosen = frozenset(selected)
    additions = set()
    if len(chosen) < 4:
        for antecedent in valid_antecedents:
            if len(antecedent) == len(chosen) + 1 and chosen.issubset(antecedent):
                additions.update(antecedent - chosen)
    return [x for x in ordered_items if x in chosen or x in additions]


def explain_rule(antecedent, target, group_count, joint_count, population, baseline, label):
    conditions = '; '.join(label(x) for x in sorted(antecedent))
    confidence = joint_count / group_count
    support = joint_count / population
    lift = confidence / baseline
    difference = 100 * (confidence - baseline)
    return (
        f'Trong dữ liệu, có {group_count} người cùng chọn các mức trả lời: {conditions}. '
        f'Trong nhóm này, {joint_count} người cũng chọn “{label(target)}”, '
        f'tương đương {confidence:.2%}. Trong toàn bộ {population} người, '
        f'mức trả lời ở vế phải xuất hiện ở {baseline:.2%}. '
        f'Tỷ lệ trong nhóm cao hơn {difference:.2f} điểm phần trăm, '
        f'tương đương {lift:.2f} lần mức nền. '
        f'Cả tổ hợp xuất hiện ở {support:.2%} toàn mẫu. '
        'Đây là liên hệ giữa các câu trả lời trong mẫu; không có nghĩa các đặc điểm gây ra nhau '
        'hoặc chắc chắn đúng với một người mới.'
    )
