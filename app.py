from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw_retail_sales.csv"
NUMERIC_COLUMNS = [
    "customer_age", "quantity", "unit_price_egp", "discount_pct",
    "sales_egp", "cost_egp", "profit_egp", "shipping_days",
]
TEXT_COLUMNS = [
    "gender", "city", "district", "store_id", "sales_channel", "category",
    "product_name", "payment_method", "returned",
]
EGP_COLUMNS = ["unit_price_egp", "sales_egp", "cost_egp", "profit_egp"]
PALETTE = ["#61d6c4", "#8c7bff", "#ffb86b", "#ff6f91", "#70a7ff", "#a7dc75"]
CANONICAL_TEXT = {
    "gender": {"female": "Female", "f": "Female", "male": "Male", "m": "Male"},
    "city": {name.casefold(): name for name in ["Alexandria", "Aswan", "Cairo", "Giza", "Mansoura", "Tanta"]},
    "district": {name.casefold(): name for name in ["6th of October", "Aswan", "Dokki", "Haram", "Heliopolis", "Maadi", "Mansoura", "Miami", "Nasr City", "New Cairo", "Sidi Gaber", "Smouha", "Tanta"]},
    "sales_channel": {name.casefold(): name for name in ["Marketplace", "Online", "Store"]},
    "returned": {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "no": "No", "n": "No", "false": "No", "0": "No"},
}

st.set_page_config(page_title="نبض المبيعات | تحليل تجزئة", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], [data-testid="stMarkdownContainer"] { font-family: 'Cairo', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 15% 0%, #17233a 0, #0b1020 40%, #080c17 100%); }
    [data-testid="stSidebar"] { background: #0c1424; border-left: 1px solid #26334a; }
    h1, h2, h3, p, label { text-align: right; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #283750; border-radius: 22px;
            background: linear-gradient(120deg, rgba(28,43,68,.96), rgba(17,25,43,.92)); margin-bottom: 1.1rem; }
    .hero-kicker { color: #61d6c4; font-size: .82rem; font-weight: 700; letter-spacing: .08em; }
    .hero h1 { margin: .35rem 0; font-size: 2rem; color: #f2f6ff; }
    .hero p { color: #aab8ce; margin: 0; }
    [data-testid="stMetric"] { background: rgba(18,28,46,.9); border: 1px solid #2a3850;
        border-radius: 16px; padding: 16px 18px; direction: rtl; }
    [data-testid="stMetricLabel"] { justify-content: flex-end; color: #aab8ce; font-size: .78rem; }
    [data-testid="stMetricValue"] { color: #f1f5ff; direction: ltr; unicode-bidi: isolate; text-align: right; }
    [data-testid="stPlotlyChart"] { border: 1px solid #243249; border-radius: 16px; background: #10192a; padding: 8px; }
    .insight { border: 1px solid #2b3b55; border-right: 4px solid #61d6c4; background: #111c2e;
               border-radius: 12px; padding: 12px 15px; margin: 8px 0; color: #e6edf8;
               direction: rtl; text-align: right; unicode-bidi: isolate; line-height: 1.9; }
    .muted { color: #96a7c0; font-size: .88rem; }
    div[data-testid="stDataFrame"] { direction: rtl; }
    @media (max-width: 1024px) {
        [data-testid="stAppViewContainer"] .main .block-container {
            padding: 1rem .75rem 2rem .75rem;
            max-width: 100%;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: .65rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
        .st-key-kpi-row [data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: wrap !important;
        }
        .st-key-kpi-row [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 calc(50% - .4rem) !important;
            width: calc(50% - .4rem) !important;
            min-width: calc(50% - .4rem) !important;
        }
        .hero { padding: 1rem; border-radius: 16px; }
        .hero h1 { font-size: 1.55rem; }
        [data-testid="stMetric"] { padding: 12px 14px; min-height: 84px; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        [data-testid="stMetricLabel"] { font-size: .76rem; }
        [data-testid="stPlotlyChart"] { padding: 2px; }
        [data-testid="stPlotlyChart"] .modebar-container { display: none !important; }
        [data-testid="stPlotlyChart"] .js-plotly-plot { touch-action: pan-y; }
        .insight { padding: 10px 12px; font-size: .94rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="بنقرأ البيانات وبنظفها...")
def load_and_clean(path: str, modified: float) -> tuple[pd.DataFrame, dict]:
    del modified  # included in cache key so a changed CSV is reloaded
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = raw.copy()
    for col in TEXT_COLUMNS:
        if col in df:
            values = (df[col].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
                      .replace({"": pd.NA, "N/A": pd.NA, "n/a": pd.NA, "NA": pd.NA}))
            # Merge capitalization variants using the most common spelling.
            frequency = values.dropna().value_counts()
            spellings: dict[str, str] = {}
            for value, count in frequency.items():
                key = value.casefold()
                if key not in spellings or count > frequency.get(spellings[key], 0):
                    spellings[key] = value
            values = values.map(lambda value: spellings.get(value.casefold(), value) if pd.notna(value) else value)
            canonical = CANONICAL_TEXT.get(col, {})
            df[col] = values.map(lambda value: canonical.get(value.casefold(), value) if pd.notna(value) else value)
    for col in NUMERIC_COLUMNS:
        if col in df:
            series = df[col].astype(str).str.replace("\u00a0", "", regex=False).str.strip()
            if col in EGP_COLUMNS:
                series = series.str.replace(r"(?i)\s*EGP\s*", "", regex=True).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    raw_count = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    before = df[NUMERIC_COLUMNS].isna().sum().sum()

    # Business rules: invalid placeholders and impossible values become missing.
    if "customer_age" in df:
        df.loc[(df.customer_age <= 0) | (df.customer_age > 100) | (df.customer_age == 9999), "customer_age"] = pd.NA
    if "quantity" in df:
        df.loc[(df.quantity <= 0) | (df.quantity >= 9999), "quantity"] = pd.NA
    if "unit_price_egp" in df:
        df.loc[(df.unit_price_egp <= 0) | (df.unit_price_egp >= 9999), "unit_price_egp"] = pd.NA
    if "discount_pct" in df:
        df.loc[(df.discount_pct < 0) | (df.discount_pct > 1), "discount_pct"] = pd.NA
    if "shipping_days" in df:
        df.loc[(df.shipping_days <= 0) | (df.shipping_days >= 9999), "shipping_days"] = pd.NA
    for col in ["sales_egp", "cost_egp"]:
        if col in df:
            df.loc[df[col] < 0, col] = pd.NA

    # Reconcile derived amounts when their source fields are available.
    inputs = df[["quantity", "unit_price_egp", "discount_pct"]].notna().all(axis=1)
    df.loc[inputs, "sales_egp"] = (
        df.loc[inputs, "quantity"] * df.loc[inputs, "unit_price_egp"] *
        (1 - df.loc[inputs, "discount_pct"])
    ).round(2)
    costs_available = df[["sales_egp", "cost_egp"]].notna().all(axis=1)
    df.loc[costs_available, "profit_egp"] = (
        df.loc[costs_available, "sales_egp"] - df.loc[costs_available, "cost_egp"]
    ).round(2)
    df["profit_margin_pct"] = (df["profit_egp"] / df["sales_egp"].where(df["sales_egp"] != 0) * 100)
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()

    missing_count = int(df[NUMERIC_COLUMNS].isna().sum().sum())
    quality = {
        "raw_rows": raw_count,
        "clean_rows": len(df),
        "duplicates_removed": duplicate_count,
        "missing_numeric_after_cleaning": missing_count,
        "numeric_missing_added_by_rules": max(0, missing_count - int(before)),
    }
    return df, quality


def egp(value: float) -> str:
    return f"{value:,.0f} ج.م"


def chart_layout(fig: go.Figure, height: int = 350) -> go.Figure:
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Cairo, sans-serif", color="#dce7f7"),
        margin=dict(l=18, r=18, t=48, b=16), legend=dict(orientation="h", y=1.12, x=1, xanchor="right", itemclick=False, itemdoubleclick=False),
        hoverlabel=dict(bgcolor="#142238", font_size=13, font_family="Cairo"),
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(137,162,195,.14)", zeroline=False)
    return fig


def build_local_insights(data: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if data.empty:
        return ["مفيش بيانات كفاية للفلتر الحالي. جرّب توسّع الاختيارات."]
    revenue = data["sales_egp"].sum(min_count=1)
    profit = data["profit_egp"].sum(min_count=1)
    margin = (profit / revenue * 100) if pd.notna(revenue) and revenue else 0
    notes.append(f"هامش الربح الإجمالي في الاختيار الحالي <bdi dir='ltr'>{margin:.1f}%</bdi>. راقب الخصومات والتكلفة في الفئات الأقل هامشًا.")

    by_category = data.groupby("category", dropna=True).agg(revenue=("sales_egp", "sum"), profit=("profit_egp", "sum"), orders=("order_id", "nunique"))
    by_category = by_category[by_category.orders >= 10]
    if not by_category.empty:
        by_category["margin"] = by_category.profit / by_category.revenue.replace(0, pd.NA) * 100
        best = by_category.margin.idxmax()
        worst = by_category.margin.idxmin()
        if best != worst:
            best_label, worst_label = escape(str(best)), escape(str(worst))
            best_margin = by_category.loc[best, "margin"]
            notes.append(f"فئة <bdi dir='ltr'><strong>{best_label}</strong></bdi> هي الأعلى في هامش الربح (<bdi dir='ltr'><strong>{best_margin:.1f}%</strong></bdi>). جرّب زيادة ظهورها في العروض. أما فئة <bdi dir='ltr'><strong>{worst_label}</strong></bdi> فهي الأقل هامشًا؛ راجع أسعارها وتكاليفها لتحسين ربحيتها.")

    if "returned" in data and data["returned"].notna().any():
        returned = data["returned"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
        return_rate = returned.mean() * 100
        if return_rate >= 8:
            notes.append(f"معدل المرتجعات <bdi dir='ltr'>{return_rate:.1f}%</bdi>؛ افحص المنتجات أو المناطق التي ترفع المرتجعات قبل زيادة الإنفاق التسويقي.")
        else:
            notes.append(f"معدل المرتجعات <bdi dir='ltr'>{return_rate:.1f}%</bdi> ضمن النطاق الحالي؛ حافظ على متابعة المرتجعات حسب المنتج والقناة.")

    channel = data.groupby("sales_channel")["profit_egp"].sum(min_count=1).dropna()
    if len(channel) > 1:
        best_channel = channel.idxmax()
        notes.append(f"قناة {best_channel} صاحبة أعلى ربح إجمالي في الفلتر. قارن هامشها وتكلفة اكتساب العميل قبل تحويل الميزانية إليها.")
    return notes[:4]


def ask_openai(data: pd.DataFrame) -> str | None:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        api_key = ""
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        summary = {
            "orders": int(data.order_id.nunique()),
            "revenue_egp": round(float(data.sales_egp.sum()), 2),
            "profit_egp": round(float(data.profit_egp.sum()), 2),
            "avg_discount_pct": round(float(data.discount_pct.mean() * 100), 2),
            "top_categories_by_profit": data.groupby("category").profit_egp.sum().nlargest(5).round(2).to_dict(),
            "profit_by_channel": data.groupby("sales_channel").profit_egp.sum().round(2).to_dict(),
            "return_rate_pct": round(float(data.returned.astype(str).str.lower().isin(["yes", "true", "1"]).mean() * 100), 2),
        }
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            instructions=("أنت محلل أعمال. اكتب بالعربية المصرية المبسطة. قدم 3 توصيات عملية لزيادة الربح "
                          "استنادًا إلى الأرقام فقط. اذكر الدليل والافتراض أو تجربة صغيرة لكل توصية. لا تخترع أسبابًا."),
            input=json.dumps(summary, ensure_ascii=False),
        )
        return response.output_text
    except Exception as exc:  # keep dashboard usable even if the optional API is unavailable
        return f"تعذر إنشاء الملخص الذكي حاليًا ({type(exc).__name__}). استخدم التوصيات المحسوبة محليًا أدناه."


if not DATA_PATH.exists():
    st.error(f"ملف البيانات غير موجود: {DATA_PATH}")
    st.stop()

df, quality = load_and_clean(str(DATA_PATH), DATA_PATH.stat().st_mtime)

st.markdown(
    "<div class='hero'><div class='hero-kicker'>RETAIL SALES INTELLIGENCE</div>"
    "<h1>نبض المبيعات</h1><p>افهم أداء الفروع والمنتجات، واكتشف فرص نمو الربح من مكان واحد.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎛️ فلاتر التحليل")
    st.caption("كل اختيار هنا يحدّث المؤشرات والرسومات والتوصيات مباشرة.")
    filtered = df.copy()
    filter_columns = [
        ("city", "المدينة"), ("district", "المنطقة"), ("store_id", "الفرع"),
        ("category", "الفئة"), ("product_name", "المنتج"),
        ("sales_channel", "قناة البيع"), ("gender", "النوع"),
        ("payment_method", "الدفع"), ("returned", "المرتجعات"),
    ]
    for column, label in filter_columns:
        options = sorted(df[column].dropna().astype(str).unique().tolist())
        selected = st.multiselect(label, options, key=f"filter_{column}")
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]
    valid_dates = df.order_date.dropna()
    if not valid_dates.empty:
        date_range = st.date_input("فترة الطلب", value=(valid_dates.min().date(), valid_dates.max().date()), key="date_range")
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            filtered = filtered[filtered.order_date.between(start, end)]
    st.divider()
    page = st.radio("الصفحة", ["لوحة الأداء", "رؤى وتوصيات"], label_visibility="collapsed")
    st.caption(f"البيانات بعد الفلاتر: {len(filtered):,} سجل")

if filtered.empty:
    st.warning("مفيش سجلات مطابقة للفلاتر الحالية. غيّر الاختيارات من القائمة الجانبية.")
    st.stop()

orders = int(filtered.order_id.nunique())
revenue = float(filtered.sales_egp.sum(min_count=1) or 0)
profit = float(filtered.profit_egp.sum(min_count=1) or 0)
margin = profit / revenue * 100 if revenue else 0
avg_order = revenue / orders if orders else 0
return_rate = filtered.returned.astype(str).str.strip().str.lower().isin(["yes", "true", "1"]).mean() * 100

if page == "لوحة الأداء":
    with st.container(key="kpi-row"):
        a, b, c, d, e = st.columns(5)
        a.metric("المبيعات · مليون", f"{revenue / 1_000_000:,.1f}", help="مليون جنيه مصري")
        b.metric("الربح · مليون", f"{profit / 1_000_000:,.1f}", help="مليون جنيه مصري")
        c.metric("هامش الربح", f"{margin:.1f}%")
        d.metric("الطلبات · ألف", f"{orders / 1_000:,.1f}", help="ألف طلب")
        e.metric("متوسط الطلب", f"{avg_order:,.0f}", help="جنيه مصري")

    month = filtered.dropna(subset=["order_month"]).groupby("order_month", as_index=False).agg(
        المبيعات=("sales_egp", "sum"), الربح=("profit_egp", "sum"))
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("المبيعات والربح عبر الوقت")
        month_long = month.melt(id_vars="order_month", var_name="المقياس", value_name="القيمة")
        fig = px.line(month_long, x="order_month", y="القيمة", color="المقياس", markers=True, color_discrete_sequence=PALETTE)
        fig.update_traces(line_width=3, mode="lines+markers")
        fig.update_layout(xaxis_title="الشهر", yaxis_title="جنيه مصري", hovermode="x unified", legend_title_text="")
        st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displaylogo": False})
    with right:
        st.subheader("المبيعات حسب قناة البيع")
        channel = filtered.groupby("sales_channel", as_index=False).agg(المبيعات=("sales_egp", "sum"), الطلبات=("order_id", "nunique"))
        fig = px.pie(channel, names="sales_channel", values="المبيعات", hole=.62, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>%{value:,.0f} ج.م<extra></extra>")
        st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displaylogo": False})

    left, right = st.columns(2)
    with left:
        st.subheader("أداء المدن")
        city = filtered.groupby("city", as_index=False).agg(المبيعات=("sales_egp", "sum"), الربح=("profit_egp", "sum"))
        fig = px.bar(city.sort_values("المبيعات", ascending=True), x="المبيعات", y="city", orientation="h", color="الربح", color_continuous_scale="Tealgrn", hover_data={"المبيعات": ":,.0f", "الربح": ":,.0f"})
        fig.update_layout(xaxis_title="جنيه مصري", yaxis_title="المدينة", coloraxis_colorbar_title="الربح")
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True, config={"displaylogo": False})
    with right:
        st.subheader("ربحية الفئات")
        category = filtered.groupby("category", as_index=False).agg(المبيعات=("sales_egp", "sum"), الربح=("profit_egp", "sum"))
        fig = px.bar(category.sort_values("الربح", ascending=True), x="الربح", y="category", orientation="h", color="المبيعات", color_continuous_scale="Purples", hover_data={"المبيعات": ":,.0f", "الربح": ":,.0f"})
        fig.update_layout(xaxis_title="جنيه مصري", yaxis_title="الفئة", coloraxis_colorbar_title="المبيعات")
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True, config={"displaylogo": False})

    st.subheader("أفضل المنتجات حسب الربح")
    products = filtered.groupby("product_name", as_index=False).agg(الربح=("profit_egp", "sum"), المبيعات=("sales_egp", "sum"), الطلبات=("order_id", "nunique")).nlargest(10, "الربح")
    fig = px.bar(products.sort_values("الربح"), x="الربح", y="product_name", orientation="h", color="الربح", color_continuous_scale="Mint", hover_data={"الطلبات": True, "المبيعات": ":,.0f"})
    fig.update_layout(xaxis_title="جنيه مصري", yaxis_title="المنتج", coloraxis_showscale=False)
    st.plotly_chart(chart_layout(fig, 430), use_container_width=True, config={"displaylogo": False})

    with st.expander("معاينة البيانات بعد التنظيف"):
        st.caption(f"اتشال {quality['duplicates_removed']:,} صف مكرر من {quality['raw_rows']:,} صف خام. الأعمدة الرقمية بها {quality['missing_numeric_after_cleaning']:,} قيمة ناقصة/غير صالحة.")
        st.dataframe(filtered.drop(columns=["order_month"], errors="ignore"), use_container_width=True, hide_index=True)
        st.download_button("تنزيل البيانات بعد التنظيف", filtered.to_csv(index=False).encode("utf-8-sig"), "clean_sales_filtered.csv", "text/csv")
else:
    st.subheader("فرص عملية لتحسين الربح")
    st.markdown(f"<p class='muted'>التحليل مبني على {orders:,} طلبًا بعد تطبيق الفلاتر الحالية. معدل المرتجعات: {return_rate:.1f}%.</p>", unsafe_allow_html=True)
    notes = build_local_insights(filtered)
    for note in notes:
        st.markdown(f"<div class='insight'>💡 {note}</div>", unsafe_allow_html=True)
    st.markdown("### ملخص الذكاء الاصطناعي")
    st.caption("يرسل الملخص أرقامًا مجمعة فقط إلى OpenAI؛ لا تُرسل بيانات العملاء أو تفاصيل الطلبات.")
    if st.button("أنشئ توصيات ذكية من الأرقام الحالية", type="primary"):
        with st.spinner("بنحلل المؤشرات..."):
            ai_text = ask_openai(filtered)
        if ai_text:
            st.markdown(ai_text)
        else:
            st.info("التوصيات المحلية جاهزة فوق. لتفعيل الملخص التوليدي، أضف OPENAI_API_KEY في Streamlit Secrets أو كمتغير بيئة.")
    st.markdown("### فحص جودة البيانات")
    q1, q2, q3 = st.columns(3)
    q1.metric("الصفوف الخام", f"{quality['raw_rows']:,}")
    q2.metric("صفوف مكررة أُزيلت", f"{quality['duplicates_removed']:,}")
    q3.metric("سجلات بعد إزالة التكرار", f"{quality['clean_rows']:,}")

