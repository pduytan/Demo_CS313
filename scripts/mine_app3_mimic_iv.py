import sys, time, statistics
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

# ================= CẤU HÌNH =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "app3_mimic_iv"
OUT_CSV = PROJECT_ROOT / "results" / "app3_mimic_iv" / "comorbidity_rules.csv"
MIN_SUP, MIN_CONF, MAX_LEN = 0.10, 0.60, 4
# Mã KHÔNG phải bệnh: ICD-9 nhóm V (yếu tố ảnh hưởng sức khỏe), E (nguyên nhân bên ngoài);
# ICD-10 nhóm R (triệu chứng), V-W-X-Y (nguyên nhân bên ngoài), Z (yếu tố ảnh hưởng sức khỏe).
# Lưu ý: ICD-10 nhóm E là bệnh nội tiết/chuyển hóa (đái tháo đường, rối loạn mỡ máu) nên PHẢI giữ lại.
NOT_DISEASE = r"^(?:9_[VE]|10_[RVWXYZ])"
# ============================================

def sep(t): print(f"\n===== {t} =====")

# --- 1. Đọc dữ liệu và tạo mã nhóm 3 ký tự (kèm phiên bản ICD) ---
dx    = pd.read_csv(f"{ROOT}/hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
d_icd = pd.read_csv(f"{ROOT}/hosp/d_icd_diagnoses.csv.gz", dtype={"icd_code": str})
for t in (dx, d_icd):
    t["icd_code"] = t["icd_code"].str.strip()
    t["icd3"] = t["icd_version"].astype(str) + "_" + t["icd_code"].str[:3]
n_before = len(dx)
dx = dx[~dx["icd3"].str.contains(NOT_DISEASE, regex=True)]
print(f"Loại {n_before - len(dx)} dòng không phải bệnh, còn {len(dx)} dòng")

# Tên đại diện cho mỗi nhóm = tên của mã đầy đủ phổ biến nhất trong nhóm
top = (dx.groupby(["icd3", "icd_version", "icd_code"]).size().reset_index(name="n")
         .sort_values("n", ascending=False).drop_duplicates("icd3")
         .merge(d_icd[["icd_code", "icd_version", "long_title"]], on=["icd_code", "icd_version"], how="left"))
names = top.set_index("icd3")["long_title"].to_dict()
fmt = lambda s: ", ".join(names.get(i, i) for i in s)

# --- 2. Mỗi bệnh nhân = 1 giao dịch ---
transactions = dx.groupby("subject_id")["icd3"].apply(lambda s: sorted(set(s))).tolist()
te = TransactionEncoder()
df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

# --- 3. Thống kê dữ liệu (mục 4.1) ---
sep("Thống kê dữ liệu")
print("Số bệnh nhân:", dx["subject_id"].nunique())
print("Số dòng chẩn đoán:", len(dx))
print("Số mã 3 ký tự:", dx["icd3"].nunique())
print(dx["icd_version"].value_counts().to_string())
print(dx.groupby("subject_id")["icd3"].nunique().describe().round(1).to_string())

# --- 4. Chạy chính: FP-Growth + luật ---
def mine(min_sup, min_conf):
    freq = fpgrowth(df, min_support=min_sup, use_colnames=True, max_len=MAX_LEN)
    try:
        r = association_rules(freq, metric="confidence", min_threshold=min_conf)
        r = r[r["lift"] > 1].sort_values("lift", ascending=False)
    except ValueError:                     # không có tập đủ lớn để sinh luật
        r = pd.DataFrame()
    return freq, r

sep(f"Kết quả chính (min_support={MIN_SUP}, min_confidence={MIN_CONF})")
freq, rules = mine(MIN_SUP, MIN_CONF)
print("Số tập phổ biến:", len(freq), "| Số luật:", len(rules))
print(freq.sort_values("support", ascending=False).head(10).to_string())
rules["antecedents"] = rules["antecedents"].map(fmt)
rules["consequents"] = rules["consequents"].map(fmt)
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
rules[cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
pd.set_option("display.max_colwidth", 70); pd.set_option("display.width", 250)
print(rules[rules["support"] >= 0.15][cols].head(15).round(2).to_string())

# --- 5. So sánh Apriori và FP-Growth (mục 4.2): chạy 5 lần, lấy trung vị ---
def timed(fn, s, reps=5):
    ts, res = [], None
    for _ in range(reps):
        t = time.perf_counter(); res = fn(df, min_support=s, use_colnames=True, max_len=MAX_LEN)
        ts.append(time.perf_counter() - t)
    return res, statistics.median(ts)

sep("Apriori vs FP-Growth")
for s in [0.20, 0.15, 0.10, 0.08]:
    a, ta = timed(apriori, s); f, tf = timed(fpgrowth, s)
    same = set(a["itemsets"]) == set(f["itemsets"])
    print(f"support={s}: Apriori {len(a)} tập, {ta:.4f}s | FP-Growth {len(f)} tập, {tf:.4f}s | cùng kết quả: {same}")

# --- 6. Thử nhiều tham số: số luật theo (min_support x min_confidence) ---
sep("Số luật theo min_support (hàng) và min_confidence (cột)")
sups, confs = [0.08, 0.10, 0.15, 0.20], [0.5, 0.6, 0.7, 0.8]
grid = pd.DataFrame({c: [len(mine(s, c)[1]) for s in sups] for c in confs}, index=sups)
grid.index.name = "support \\ conf"
print(grid.to_string())
