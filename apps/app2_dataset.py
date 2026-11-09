"""Explicit encoding of all 150 source columns; original data is preserved."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'app2_young_people_survey'
NUMERIC = ['Age', 'Height', 'Weight', 'Number of siblings']


def prepare(mode='exact'):
    if mode not in ['exact', 'grouped']:
        raise ValueError('mode must be exact or grouped')
    raw = pd.read_csv(DATA_DIR / 'responses.csv')
    # Complete-case policy, same principle as the previous 24-column analysis.
    complete = raw.loc[~raw['Height'].eq(62)].dropna().copy()
    encoded = pd.DataFrame(index=complete.index)
    dictionary = []
    for col in raw.columns:
        values = complete[col]
        if col in NUMERIC:
            # Empirical thirds are exploratory bins, not medical/physical standards.
            quantiles = values.quantile([1 / 3, 2 / 3]).tolist()
            if col == 'Number of siblings':
                categories = pd.cut(
                    values, [-float('inf'), 0, 1, float('inf')],
                    labels=['0', '1', '2+'],
                )
                rule = '0; 1; 2+ siblings'
            else:
                edges = sorted(set([-float('inf'), *quantiles, float('inf')]))
                labels = [f'({edges[i]:.3g}, {edges[i+1]:.3g}]' for i in range(len(edges) - 1)]
                categories = pd.cut(values, edges, labels=labels)
                rule = 'Empirical tertiles on complete cases; exploratory, not normative'
        elif pd.api.types.is_numeric_dtype(raw[col]):
            if not values.isin([1, 2, 3, 4, 5]).all():
                raise ValueError(f'Unexpected scale: {col}')
            if mode == 'exact':
                categories = pd.Series(
                    pd.Categorical(values.astype(int).astype(str), categories=['1', '2', '3', '4', '5']),
                    index=values.index,
                )
                rule = 'Exact question-answer item: retain each original score 1,2,3,4,5'
            else:
                categories = pd.cut(values, [0, 2, 3, 5], labels=['1-2', '3', '4-5'])
                rule = 'Score 1-2; score 3; score 4-5 (interpret with original question)'
        else:
            categories = values.astype(str)
            rule = 'Original categorical response'
        dummies = pd.get_dummies(categories, prefix=col, prefix_sep=' = ', dtype=bool)
        encoded = pd.concat([encoded, dummies], axis=1)
        dictionary.append({
            'column': col,
            'encoding': rule,
            'possible_items': len(dummies.columns),
            'q33': quantiles[0] if col in NUMERIC and col != 'Number of siblings' else None,
            'q67': quantiles[1] if col in NUMERIC and col != 'Number of siblings' else None,
        })
    assert len(dictionary) == 150
    assert encoded.sum(axis=1).eq(150).all()
    return raw, encoded, pd.DataFrame(dictionary)
