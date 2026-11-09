# FP-Growth trong kinh doanh: dữ liệu -> EDA -> cấu hình -> kết quả.
# Giải nén 01_Du_lieu_Online_Retail.zip cùng thư mục với tệp này.
# Cài: python -m pip install pandas numpy openpyxl mlxtend matplotlib
# Chạy: python 03_FP_Growth_Kinh_doanh.py
# Kết quả lưu trong results/ và figures/. Nguồn dữ liệu UCI: DOI 10.24432/C5BW33.
# Các phần mã sau là những thành phần đã dùng tạo báo cáo, ghép để chạy bằng một lệnh.


# Tải dữ liệu gốc từ UCI nếu chưa có; bản CSV sạch trong ZIP dùng ngay cho demo.
from pathlib import Path
import urllib.request, zipfile, hashlib
_input_root=Path(__file__).resolve().parent/'data'
_input_root.mkdir(exist_ok=True)
_raw_workbook=_input_root/'Online Retail.xlsx'
if not _raw_workbook.exists():
    print('Downloading original Online Retail dataset from UCI (about 24 MB)...', flush=True)
    _archive=_input_root/'online_retail_uci.zip'
    urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/352/online+retail.zip',_archive)
    with zipfile.ZipFile(_archive) as _z:
        _raw_workbook.write_bytes(_z.read('Online Retail.xlsx'))
    if hashlib.sha256(_raw_workbook.read_bytes()).hexdigest()!='43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d':
        raise ValueError('The source workbook differs from the report reference; verify the UCI release before comparing counts.')


# ===== analysis.py =====
"""Reproduce the business case study from the original UCI workbook.

Install pandas, numpy, openpyxl, mlxtend and matplotlib in your environment.
Run: python analysis.py
"""
from pathlib import Path
import sys
import json
import math
import hashlib
import platform
from collections import Counter

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT.parents[1] / 'tmp' / 'fpgrowth_vendor'
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
import numpy as np
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import fpgrowth, association_rules

sys.stdout.reconfigure(encoding='utf-8')
RESULT = ROOT / 'results'
RESULT.mkdir(exist_ok=True)
source = ROOT / 'data' / 'Online Retail.xlsx'
cache = ROOT / 'data' / 'online_retail_raw.pkl'
if cache.exists():
    raw = pd.read_pickle(cache)
else:
    raw = pd.read_excel(source, engine='openpyxl', dtype={'InvoiceNo': str, 'StockCode': str})
    raw.to_pickle(cache)
raw['InvoiceNo'] = raw['InvoiceNo'].astype(str).str.strip().str.upper()
raw['StockCode'] = raw['StockCode'].astype(str).str.strip().str.upper()
raw['InvoiceDate'] = pd.to_datetime(raw['InvoiceDate'])
stats = {'source_rows': len(raw), 'source_columns': list(raw.columns),
         'source_date_min': str(raw.InvoiceDate.min()), 'source_date_max': str(raw.InvoiceDate.max()),
         'workbook_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
         'run_date': str(pd.Timestamp.today().date()), 'python': platform.python_version(),
         'pandas': pd.__version__, 'numpy': np.__version__, 'mlxtend': mlxtend.__version__}
stages = [{'step': 'Dữ liệu gốc toàn bộ quốc gia', 'rows': len(raw)}]
d = raw.loc[raw.Country.eq('United Kingdom')].copy()
stages.append({'step': 'Chỉ thị trường United Kingdom', 'rows': len(d)})
d = d.loc[~d.InvoiceNo.str.startswith('C')].copy()
stages.append({'step': 'Loại hóa đơn hủy', 'rows': len(d)})
d = d.loc[d.Quantity.gt(0)].copy()
stages.append({'step': 'Giữ số lượng dương', 'rows': len(d)})
d = d.loc[d.UnitPrice.gt(0)].copy()
stages.append({'step': 'Giữ đơn giá dương', 'rows': len(d)})
is_product = d.StockCode.str.fullmatch(r'\d{5}[A-Z]{0,2}', na=False)
stats['excluded_codes'] = d.loc[~is_product, 'StockCode'].value_counts().to_dict()
d = d.loc[is_product].copy()
stages.append({'step': 'Giữ mã hàng hóa năm chữ số và tối đa hai chữ cái', 'rows': len(d)})
label = d.dropna(subset=['Description']).copy()
label['Description'] = label.Description.str.strip()
label = label.groupby('StockCode').Description.agg(lambda s: s.value_counts().index[0]).to_dict()
dates = d.groupby('InvoiceNo').InvoiceDate.min()
pairs = d[['InvoiceNo', 'StockCode']].drop_duplicates()
stages.append({'step': 'Một lần hiện diện cho mỗi sản phẩm trong mỗi hóa đơn', 'rows': len(pairs)})
baskets = pairs.groupby('InvoiceNo').StockCode.agg(list)
n = len(baskets)
freq = Counter(x for row in baskets for x in row)
top = pd.DataFrame([{'code': k, 'name': label.get(k, k), 'count': v, 'support': v/n} for k,v in freq.items()])
top = top.sort_values(['count','code'], ascending=[False,True]).reset_index(drop=True)
top.to_csv(RESULT/'product_support.csv', index=False, encoding='utf-8-sig')
stats.update({'stages': stages, 'clean_line_rows': len(d), 'basket_item_pairs': len(pairs),
              'n_baskets': n, 'n_products': len(freq), 'missing_customer_rows_kept': int(d.CustomerID.isna().sum()),
              'basket_size_median': float(baskets.map(len).median()),
              'basket_size_mean': float(baskets.map(len).mean()),
              'n_single_item_baskets': int(baskets.map(len).eq(1).sum()),
              'date_min': str(d.InvoiceDate.min()), 'date_max': str(d.InvoiceDate.max()),
              'main_minsup': .02, 'main_minconf': .60, 'main_minlift': 1.5, 'max_len': 3,
              'min_count_main': math.ceil(.02*n)})
# Only globally infrequent items below the smallest tested threshold are removed.
# Retain ALL invoices, including those left without selected items, in the denominator.
codes = sorted(k for k,v in freq.items() if v/n >= .005)
idx = {k:i for i,k in enumerate(codes)}
arr = np.zeros((n,len(codes)), dtype=bool)
for i,row in enumerate(baskets):
    for x in row:
        if x in idx:
            arr[i,idx[x]] = True
frame = pd.DataFrame(arr, index=baskets.index, columns=codes)

def serialize_rules(rules, name):
    out = rules.copy()
    out['antecedents'] = out.antecedents.map(lambda s: '|'.join(sorted(s)))
    out['consequents'] = out.consequents.map(lambda s: '|'.join(sorted(s)))
    out.to_csv(RESULT/name, index=False, encoding='utf-8-sig')

sensitivity = []
main_fi = main_rules = None
for minsup in [.005,.01,.02,.03,.05]:
    fi = fpgrowth(frame, min_support=minsup, use_colnames=True, max_len=3)
    rules = association_rules(fi, num_itemsets=n, metric='confidence', min_threshold=.6)
    good = rules.loc[rules.lift.ge(1.5)].copy()
    counts = fi.itemsets.map(len).value_counts().to_dict()
    sensitivity.append({'minsup':minsup, 'min_count':math.ceil(minsup*n),
                        'itemsets':len(fi), 'singletons':counts.get(1,0), 'pairs':counts.get(2,0),
                        'triples':counts.get(3,0), 'rules':len(good)})
    print('THRESHOLD', sensitivity[-1], flush=True)
    if minsup == .02:
        main_fi, main_rules = fi.copy(), good.copy()
        serialize_rules(rules, 'rules_confidence60_before_lift_filter.csv')
assert main_fi is not None
pd.DataFrame(sensitivity).to_csv(RESULT/'sensitivity.csv',index=False,encoding='utf-8-sig')
stats['sensitivity'] = sensitivity
stats['n_itemsets'] = len(main_fi)
stats['n_rules'] = len(main_rules)
fi_out = main_fi.copy()
fi_out['itemsets'] = fi_out.itemsets.map(lambda s:'|'.join(sorted(s)))
fi_out['count'] = np.rint(fi_out.support*n).astype(int)
fi_out.to_csv(RESULT/'frequent_itemsets.csv',index=False,encoding='utf-8-sig')
main_rules = main_rules.sort_values(['support','confidence','lift'], ascending=False).reset_index(drop=True)
serialize_rules(main_rules, 'association_rules.csv')

# Select single-product rules with substantial joint support. Keep one direction
# per unordered pair to avoid presenting reversed copies as independent findings.
single_rules = main_rules.loc[main_rules.antecedents.map(len).eq(1)&main_rules.consequents.map(len).eq(1)]
selected = []
used_pairs = set()
for _,r in single_rules.iterrows():
    a,b=next(iter(r.antecedents)),next(iter(r.consequents))
    key=tuple(sorted([a,b]))
    if key in used_pairs:
        continue
    used_pairs.add(key)
    # Direct independent recount from original transaction sets.
    na=sum(a in row for row in baskets)
    nb=sum(b in row for row in baskets)
    nab=sum(a in row and b in row for row in baskets)
    assert abs(nab/n-float(r.support))<1e-12
    assert abs(nab/na-float(r.confidence))<1e-12
    assert abs((nab*n)/(na*nb)-float(r.lift))<1e-10
    selected.append({'a':a,'b':b,'a_name':label[a],'b_name':label[b],
                     'n_a':na,'n_b':nb,'n_ab':nab,'support':nab/n,'confidence':nab/na,
                     'reverse_confidence':nab/nb,'lift':(nab*n)/(na*nb)})
    if len(selected)==8:
        break
stats['selected_rules']=selected
stats['report_rules']=[selected[i] for i in [0,1,5]]

# Temporal check: discover candidates only before October, then recount them on
# October-December invoices. Test rules are never used to fit the train FP-tree.
cutoff=pd.Timestamp('2011-10-01')
train_ids=dates[dates.lt(cutoff)].index.intersection(frame.index)
test_ids=dates[dates.ge(cutoff)].index.intersection(frame.index)
assert len(train_ids.intersection(test_ids)) == 0
assert len(train_ids)+len(test_ids)==n
tr=frame.loc[train_ids]
te=frame.loc[test_ids]
train_fi=fpgrowth(tr,min_support=.02,use_colnames=True,max_len=3)
train_rules=association_rules(train_fi,num_itemsets=len(tr),metric='confidence',min_threshold=.6)
train_rules=train_rules.loc[train_rules.lift.ge(1.5)]
train_rules=train_rules.loc[train_rules.antecedents.map(len).eq(1)&train_rules.consequents.map(len).eq(1)]
train_rules=train_rules.sort_values(['support','confidence'],ascending=False)
temporal=[]
seen=set()
for _,r in train_rules.iterrows():
    a,b=next(iter(r.antecedents)),next(iter(r.consequents))
    key=tuple(sorted([a,b]))
    if key in seen: continue
    seen.add(key)
    va=te[a].to_numpy() if a in te else np.zeros(len(te),dtype=bool)
    vb=te[b].to_numpy() if b in te else np.zeros(len(te),dtype=bool)
    na,nb,nab=int(va.sum()),int(vb.sum()),int((va&vb).sum())
    temporal.append({'a':a,'b':b,'a_name':label[a],'b_name':label[b],
                     'train_support':float(r.support),'train_confidence':float(r.confidence),'train_lift':float(r.lift),
                     'test_support':nab/len(te),'test_confidence':nab/na if na else None,
                     'test_lift':nab*len(te)/(na*nb) if na*nb else None,
                     'test_n_a':na,'test_n_b':nb,'test_n_ab':nab})
    if len(temporal)==3:break
stats['temporal']={'cutoff':'2011-10-01','n_train':len(tr),'n_test':len(te),'rules':temporal}
stats['labels']=label
(RESULT/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
pairs.to_csv(RESULT/'basket_items.csv',index=False,encoding='utf-8-sig')
print(json.dumps({k:v for k,v in stats.items() if k not in ['labels','excluded_codes']},ensure_ascii=False,indent=2))


# ===== eda_business.py =====
(Path(__file__).resolve().parent/'figures').mkdir(exist_ok=True)
"""Explore the source data and the exact transaction population used in the report."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
vendor=ROOT.parents[1]/'tmp'/'fpgrowth_vendor'
if vendor.exists():sys.path.insert(0,str(vendor))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import FuncFormatter
sys.stdout.reconfigure(encoding='utf-8')
stats=json.loads((ROOT/'results/stats.json').read_text(encoding='utf-8'))
cache=ROOT/'data/online_retail_raw.pkl'
raw=pd.read_pickle(cache) if cache.exists() else pd.read_excel(ROOT/'data/Online Retail.xlsx',dtype={'InvoiceNo':str,'StockCode':str})
raw['InvoiceNo']=raw.InvoiceNo.astype(str).str.strip().str.upper()
raw['StockCode']=raw.StockCode.astype(str).str.strip().str.upper()
raw['InvoiceDate']=pd.to_datetime(raw.InvoiceDate)
d=raw.loc[raw.Country.eq('United Kingdom') & ~raw.InvoiceNo.str.startswith('C') & raw.Quantity.gt(0) & raw.UnitPrice.gt(0) & raw.StockCode.str.fullmatch(r'\d{5}[A-Z]{0,2}',na=False)].copy()
pairs=d[['InvoiceNo','StockCode']].drop_duplicates()
sizes=pairs.groupby('InvoiceNo').size()
dates=d.groupby('InvoiceNo').InvoiceDate.min().loc[sizes.index]
monthly=dates.dt.to_period('M').value_counts().sort_index()
cuts=[0,1,5,10,20,50,100,np.inf]
names=['1','2–5','6–10','11–20','21–50','51–100','Trên 100']
grouped=pd.cut(sizes,bins=cuts,labels=names).value_counts().reindex(names)
assert len(sizes)==stats['n_baskets']
assert sizes.sum()==stats['basket_item_pairs']
assert monthly.sum()==len(sizes)
assert grouped.sum()==len(sizes)
eda={
 'run_date':str(pd.Timestamp.today().date()),
 'raw_rows':len(raw),'raw_missing_by_column':{k:int(v) for k,v in raw.isna().sum().items()},
 'uk_rows_before_filter':int(raw.Country.eq('United Kingdom').sum()),
 'n_baskets':len(sizes),'n_products':int(pairs.StockCode.nunique()),
 'basket_size_mean':float(sizes.mean()),'basket_size_median':float(sizes.median()),
 'basket_size_p95':float(sizes.quantile(.95)),'basket_size_max':int(sizes.max()),
 'single_item_baskets':int(sizes.eq(1).sum()),'single_item_share':float(sizes.eq(1).mean()),
 'over100_baskets':int(sizes.gt(100).sum()),'over100_share':float(sizes.gt(100).mean()),
 'presence_density':float(len(pairs)/(len(sizes)*pairs.StockCode.nunique())),
 'monthly':{str(k):int(v) for k,v in monthly.items()},
 'size_groups':{str(k):int(v) for k,v in grouped.items()},
 'peak_month':str(monthly.idxmax()),'peak_month_invoices':int(monthly.max()),
 'partial_month':'2011-12','partial_month_last_date':str(d.InvoiceDate.max().date()),
}
(ROOT/'results/eda_stats.json').write_text(json.dumps(eda,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame({'month':monthly.index.astype(str),'invoices':monthly.values}).to_csv(ROOT/'results/eda_monthly.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'size_group':names,'invoices':grouped.values,'share':grouped.values/len(sizes)}).to_csv(ROOT/'results/eda_basket_sizes.csv',index=False,encoding='utf-8-sig')
for f in ['arial.ttf','arialbd.ttf']:
 if (Path('C:/Windows/Fonts')/f).exists():fontManager.addfont(str(Path('C:/Windows/Fonts')/f))
plt.rcParams.update({'font.family':('Arial' if Path('C:/Windows/Fonts/arial.ttf').exists() else 'DejaVu Sans'),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','text.color':'#17283D','axes.labelcolor':'#334155'})
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10.6,6.7),gridspec_kw={'height_ratios':[1,1.12]})
bars=ax1.bar(names,grouped.values,color='#173B5E',width=.64)
ax1.set_title('Số sản phẩm khác nhau trong mỗi hóa đơn',loc='left',fontweight='bold',pad=13)
ax1.set_ylabel('Số hóa đơn');ax1.set_xlabel('Số mã hàng khác nhau')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'.replace(',','.')))
ax1.set_ylim(0,grouped.max()*1.24);ax1.grid(axis='y',alpha=.13);ax1.set_axisbelow(True)
for b in bars:
 ax1.text(b.get_x()+b.get_width()/2,b.get_height()+grouped.max()*.025,f'{int(b.get_height()):,}'.replace(',','.'),ha='center',fontsize=9)
colors=['#173B5E']*len(monthly);colors[-1]='#D87533'
bars=ax2.bar(range(len(monthly)),monthly.values,color=colors,width=.65)
ax2.set_title('Số hóa đơn hợp lệ theo tháng',loc='left',fontweight='bold',pad=13)
ax2.set_xticks(range(len(monthly)),[f'{p.month:02}/{str(p.year)[2:]}' for p in monthly.index],rotation=35,ha='right')
ax2.set_ylabel('Số hóa đơn');ax2.set_xlabel('Tháng / năm')
ax2.set_ylim(0,monthly.max()*1.24);ax2.grid(axis='y',alpha=.13);ax2.set_axisbelow(True)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:,.0f}'.replace(',','.')))
for b in bars:ax2.text(b.get_x()+b.get_width()/2,b.get_height()+monthly.max()*.02,f'{int(b.get_height()):,}'.replace(',','.'),ha='center',fontsize=8.5)
fig.tight_layout(h_pad=2.8)
for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'06_eda_gio_hang_va_thoi_gian.{ext}',dpi=240,bbox_inches='tight',pad_inches=.16)
plt.close(fig)
print(json.dumps(eda,ensure_ascii=False,indent=2))


# ===== make_figures.py =====
from pathlib import Path
import sys,json,textwrap
ROOT=Path(__file__).resolve().parent
vendor=ROOT.parents[1]/'tmp'/'fpgrowth_vendor'
if vendor.exists():sys.path.insert(0,str(vendor))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties, fontManager

OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
S=json.loads((ROOT/'results'/'stats.json').read_text(encoding='utf-8'))
for f in ['arial.ttf','arialbd.ttf']:
    if (Path('C:/Windows/Fonts')/f).exists():fontManager.addfont(str(Path('C:/Windows/Fonts')/f))
plt.rcParams.update({'font.family':('Arial' if Path('C:/Windows/Fonts/arial.ttf').exists() else 'DejaVu Sans'),'font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':'#334155','text.color':'#17283D',
 'axes.titleweight':'bold','axes.titlesize':14,'figure.facecolor':'white',
 'savefig.facecolor':'white','svg.fonttype':'none'})
NAVY='#173B5E';TEAL='#0C817F';ORANGE='#D87533';GRAY='#CAD5DE'
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight',pad_inches=.17)
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',pad_inches=.17)
    plt.close(fig)

top=pd.read_csv(ROOT/'results'/'product_support.csv').head(8).iloc[::-1]
fig,ax=plt.subplots(figsize=(10.8,4.4))
labs=['\n'.join(textwrap.wrap(x,38)) for x in top.name]
bars=ax.barh(labs,top.support,color=NAVY,height=.62)
ax.set_xlim(0,top.support.max()*1.32)
ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.set_xlabel('Tỷ lệ hóa đơn có sản phẩm')
ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
for b,c,s in zip(bars,top['count'],top.support):
    label=f'{s:.1%}'.replace('.',',')+'  |  '+f'{c:,}'.replace(',','.')+' HĐ'
    ax.text(s+.002,b.get_y()+b.get_height()/2,label,va='center',fontsize=10)
fig.tight_layout();save(fig,'01_san_pham_pho_bien')

rules=S['report_rules']
x=np.arange(len(rules));w=.34
fig,ax=plt.subplots(figsize=(9.8,4.0))
base=[r['n_b']/S['n_baskets'] for r in rules]
conf=[r['confidence'] for r in rules]
a=ax.bar(x-w/2,base,w,label='Tỷ lệ mua B trên toàn bộ hóa đơn',color=GRAY)
b=ax.bar(x+w/2,conf,w,label='Tỷ lệ mua B trong các hóa đơn có A',color=TEAL)
ax.set_xticks(x,[f'R{i+1}: {r["a"]} → {r["b"]}' for i,r in enumerate(rules)])
ax.set_ylim(0,1.04);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.set_ylabel('Tỷ lệ hóa đơn');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=1,frameon=False)
for group in [a,b]:
    for bar in group:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.018,f'{bar.get_height():.1%}'.replace('.',','),ha='center',fontsize=11)
fig.tight_layout();save(fig,'02_confidence_va_muc_nen')

v=pd.DataFrame(S['sensitivity'])
v=v.loc[v.minsup.ge(.01)]
fig,ax=plt.subplots(figsize=(9.8,3.8))
ax.plot(v.minsup*100,v.itemsets,'o-',color=NAVY,label='Tập mục thường xuyên',lw=2)
ax.plot(v.minsup*100,v.rules,'s-',color=ORANGE,label='Luật đạt confidence ≥ 60% và lift ≥ 1,5',lw=2)
ax.set_xticks(v.minsup*100);ax.set_xlabel('Minimum support (%)');ax.set_ylabel('Số lượng')
ax.grid(alpha=.15);ax.legend(frameon=False)
for _,r in v.iterrows():
    ax.annotate(f'{int(r.itemsets):,}'.replace(',','.'),(r.minsup*100,r.itemsets),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
fig.tight_layout();save(fig,'03_do_nhay_minsup')

t=S['temporal']['rules'];x=np.arange(len(t))
fig,ax=plt.subplots(figsize=(9.8,3.8))
for shift,key,color,label in [(-w/2,'train_confidence',NAVY,'Trước 01/10/2011'),(w/2,'test_confidence',TEAL,'Từ 01/10 đến 09/12/2011')]:
    bars=ax.bar(x+shift,[r[key] or 0 for r in t],w,color=color,label=label)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.016,f'{bar.get_height():.1%}'.replace('.',','),ha='center',fontsize=10)
ax.set_xticks(x,[f'T{i+1}: {r["a"]} → {r["b"]}' for i,r in enumerate(t)])
ax.set_ylim(0,1.03);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.set_ylabel('Confidence');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'04_doi_chieu_thoi_gian')

fig,ax=plt.subplots(figsize=(11,2.0));ax.axis('off')
boxes=[('Hóa đơn','Một giỏ hàng / mã hóa đơn'),('FP-Growth','Tập sản phẩm thường xuyên'),('Luật kết hợp','Support · confidence · lift'),('Quyết định','Thử gợi ý mua kèm / combo')]
for i,(title,sub) in enumerate(boxes):
    xx=.015+i*.25
    ax.add_patch(FancyBboxPatch((xx,.27),.22,.57,boxstyle='round,pad=0.012,rounding_size=0.025',transform=ax.transAxes,fc='#F1F5F9',ec='#CAD5DE'))
    ax.text(xx+.11,.62,title,ha='center',va='center',weight='bold',fontsize=13,transform=ax.transAxes)
    ax.text(xx+.11,.43,sub,ha='center',va='center',fontsize=8.5,transform=ax.transAxes)
    if i<3:ax.annotate('',xy=(xx+.247,.55),xytext=(xx+.225,.55),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})
save(fig,'05_so_do_ung_dung')

equations={
 'support':r'\mathrm{support}(X)=\frac{n(X)}{N}',
 'confidence':r'\mathrm{confidence}(A\rightarrow B)=\frac{n(A\cup B)}{n(A)}',
 'lift':r'\mathrm{lift}(A\rightarrow B)=\frac{\mathrm{confidence}(A\rightarrow B)}{\mathrm{support}(B)}=\frac{N\,n(A\cup B)}{n(A)\,n(B)}',
}
for name,tex in equations.items():
    fig=plt.figure(figsize=(10,0.7));fig.text(.5,.5,f'${tex}$',ha='center',va='center',fontsize=19)
    save(fig,f'formula_{name}')
print('Figures saved to',OUT)
