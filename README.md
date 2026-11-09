# FP-Growth Association Rule Explorer (3 Applications)

Bộ demo tích hợp 3 ứng dụng khai thác tập phổ biến và luật kết hợp (**FP-Growth**) trên một giao diện Streamlit thống nhất.

---

## Cấu trúc thư mục

```text
Ung_dung_2_Demo/
├── .streamlit/
│   └── config.toml                          # Cấu hình theme giao diện chung
├── app.py                                   # Điểm khởi chạy chính & chuyển đổi 3 ứng dụng
├── run_demo.ps1                             # Script khởi chạy nhanh trên Windows PowerShell
├── requirements.txt                         # Danh sách thư viện cần thiết
├── README.md
│
├── data/                                    # Thư mục dữ liệu chung cho cả 3 ứng dụng
│   ├── app1_online_retail/                  # Dữ liệu Ứng dụng 1 (UCI Online Retail - UK)
│   │   ├── gio_hang_UK_da_xu_ly.csv
│   │   ├── ten_san_pham.json
│   │   ├── eda_so_hoa_don_theo_thang.csv
│   │   └── NGUON_DU_LIEU.txt
│   ├── app2_young_people_survey/            # Dữ liệu Ứng dụng 2 (Young People Survey)
│   │   ├── responses.csv
│   │   └── columns.csv
│   └── app3_mimic_iv/                       # Dữ liệu Ứng dụng 3 (MIMIC-IV Clinical Demo 2.2)
│       ├── hosp/
│       └── icu/
│
├── results/                                 # Tập luật kết hợp đã khai thác sẵn cho từng ứng dụng
│   ├── app1_online_retail/
│   │   └── association_rules.csv
│   ├── app2_young_people_survey/
│   │   └── association_rules.csv
│   └── app3_mimic_iv/
│       └── association_rules.csv
│
├── shared/                                  # Hàm hỗ trợ dùng chung cho cả 3 ứng dụng
│   ├── __init__.py
│   └── demo_helpers.py
│
├── apps/                                    # Giao diện và xử lý dữ liệu của từng ứng dụng
│   ├── __init__.py
│   ├── app1_online_retail.py                # Ứng dụng 1: Market Basket & Cross-Selling
│   ├── app2_young_people_survey.py          # Ứng dụng 2: Personality & Behavior Explorer
│   ├── app2_dataset.py                      # Tiền xử lý dữ liệu khảo sát Young People Survey
│   ├── app2_trait_labels.py                 # Danh mục nhãn tính cách & cảm xúc
│   └── app3_mimic_iv.py                     # Ứng dụng 3: Clinical Comorbidities Explorer
│
└── scripts/                                 # Script khai thác FP-Growth độc lập
    ├── mine_app1_online_retail.py
    └── mine_app3_mimic_iv.py
```

---

## Cách khởi chạy

Mở PowerShell tại thư mục gốc của project và chạy:

```powershell
.\run_demo.ps1
```

Hoặc chạy trực tiếp bằng lệnh:

```powershell
python -m streamlit run app.py --server.port 8503
```
