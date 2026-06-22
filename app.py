import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="个人收支流水经营分析与风险管控决策看板",
    page_icon="💸",
    layout="wide",
)

CSS = """
<style>
    :root{
        --primary:#2563eb; --primary-soft:#eff6ff; --ink:#0f172a; --muted:#64748b;
        --line:#dbeafe; --card:#ffffff; --bg:#f8fafc; --amber:#f59e0b; --red:#ef4444;
    }
    html, body, [class*="css"] {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;}
    .block-container {max-width: 1520px; padding-top: 3.4rem; padding-bottom: 3rem;}
    section[data-testid="stSidebar"] {background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%); border-right:1px solid #dbeafe;}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {color:#0f172a;}
    div[data-baseweb="tag"] {background-color:#dbeafe !important; color:#1d4ed8 !important; border:1px solid #93c5fd !important; border-radius:999px !important;}
    div[data-baseweb="tag"] span {color:#1d4ed8 !important;}
    .hero {background:linear-gradient(135deg,#ffffff 0%,#f4f8ff 58%,#f8f5ff 100%); border:1px solid #dbeafe; border-radius:24px; padding:28px 34px; box-shadow:0 18px 46px rgba(15,23,42,.06); margin-bottom:18px;}
    .hero h1 {font-size:34px; line-height:1.15; margin:0 0 10px 0; color:#0f172a; letter-spacing:-.5px;}
    .subtitle {color:#64748b; font-size:15px; margin-bottom:18px;}
    .judgement {background:#eef6ff; border-left:5px solid #2563eb; padding:16px 18px; border-radius:16px; color:#1e293b; line-height:1.85; font-size:15px;}
    .section-title {font-size:25px; font-weight:800; margin:34px 0 6px 0; color:#0f172a; letter-spacing:-.2px;}
    .section-desc {font-size:14px; color:#64748b; margin-bottom:18px;}
    .action-card {background:#ffffff; border:1px solid #e2e8f0; border-radius:18px; padding:20px 20px 18px; box-shadow:0 10px 30px rgba(15,23,42,.045); min-height:170px;}
    .action-card .tag {display:inline-block; padding:5px 10px; border-radius:999px; background:#dbeafe; color:#1d4ed8; font-weight:700; font-size:12px; margin-bottom:14px;}
    .action-card h3 {font-size:20px; margin:0 0 10px 0; color:#0f172a; line-height:1.35;}
    .action-card p {font-size:14px; color:#64748b; line-height:1.72; margin:0 0 12px 0;}
    .action-note {background:#eff6ff; border:1px solid #dbeafe; color:#1e40af; padding:10px 12px; border-radius:12px; font-size:13px; line-height:1.65;}
    .solution-strip {background:#ffffff; border:1px solid #dbeafe; border-radius:18px; padding:18px 18px; margin:18px 0 24px; box-shadow:0 10px 28px rgba(37,99,235,.04);}
    .solution-strip b {color:#1d4ed8;}
    .metric-card {background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:18px 18px; box-shadow:0 8px 24px rgba(15,23,42,.04);}
    .metric-card .label {font-size:13px; color:#64748b; margin-bottom:6px;}
    .metric-card .value {font-size:25px; font-weight:800; color:#0f172a;}
    .metric-card .hint {font-size:12px; color:#94a3b8; margin-top:5px;}
    .recommend-box {background:#eff6ff; border:1px solid #dbeafe; border-radius:14px; padding:14px 16px; color:#1e40af; line-height:1.7; font-size:14px;}
    .mini-caption {font-size:12px; color:#64748b; margin-top:-8px; margin-bottom:8px;}
    .divider {height:1px; background:#e2e8f0; margin:28px 0 10px;}
    .footer-note {font-size:12px; color:#94a3b8; margin-top:28px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"
FULL_PATH = DATA_DIR / "cashflow_risk_result_v02.csv"
FOCUS_PATH = DATA_DIR / "cashflow_focus_transactions_v02.csv"

RULES = {
    "rule_large_amount": "单笔大额",
    "rule_small_high_freq": "高频小额",
    "rule_budget_out_risk": "预算外风险",
    "rule_renewal_management": "续费管理",
    "rule_food_delivery": "外卖频次",
}

@st.cache_data
def load_data():
    full = pd.read_csv(FULL_PATH)
    focus = pd.read_csv(FOCUS_PATH)
    for df in [full, focus]:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["transaction_month"] = df["transaction_date"].dt.to_period("M").astype(str)
        for c in ["net_amount", "amount_abs", "signed_amount", "rule_hit_count"] + list(RULES.keys()):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        for c in ["category", "scene_l1", "scene_l2", "necessity_type", "budget_type", "risk_tag", "risk_level", "weekday_name", "hit_rule_names", "action_suggestion", "risk_reason"]:
            if c in df.columns:
                df[c] = df[c].fillna("未标记").astype(str)
    return full, focus

full_df, focus_df_raw = load_data()

# ---------------- Sidebar filters ----------------
st.sidebar.markdown("## 筛选器")
st.sidebar.caption("筛选后，经营诊断、风险规则、行动清单会同步变化。")

min_date = full_df["transaction_date"].min().date()
max_date = full_df["transaction_date"].max().date()
date_range = st.sidebar.date_input("交易日期", [min_date, max_date], min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) or isinstance(date_range, list):
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

def multi_filter(label, col, data):
    opts = sorted([x for x in data[col].dropna().unique().tolist() if str(x) != "nan"])
    return st.sidebar.multiselect(label, opts, default=opts)

scene_l1_sel = multi_filter("一级场景", "scene_l1", full_df)
scene_l2_sel = multi_filter("二级场景", "scene_l2", full_df)
necessity_sel = multi_filter("必要性", "necessity_type", full_df)
budget_sel = multi_filter("预算类型", "budget_type", full_df)
risk_level_sel = multi_filter("风险等级", "risk_level", full_df)

rule_options = list(RULES.values())
rule_sel = st.sidebar.multiselect("命中规则", rule_options, default=rule_options)

filtered = full_df[
    (full_df["transaction_date"] >= start_date) &
    (full_df["transaction_date"] <= end_date) &
    (full_df["scene_l1"].isin(scene_l1_sel)) &
    (full_df["scene_l2"].isin(scene_l2_sel)) &
    (full_df["necessity_type"].isin(necessity_sel)) &
    (full_df["budget_type"].isin(budget_sel)) &
    (full_df["risk_level"].isin(risk_level_sel))
].copy()

# rule filter: keep if no rule selection? If selected all, no need. If narrowed, must have at least one selected rule.
selected_rule_cols = [k for k,v in RULES.items() if v in rule_sel]
if len(selected_rule_cols) < len(RULES):
    if selected_rule_cols:
        filtered = filtered[filtered[selected_rule_cols].sum(axis=1) > 0].copy()
    else:
        filtered = filtered.iloc[0:0].copy()

# ---------------- Computations ----------------
def money(v):
    try:
        v = float(v)
    except Exception:
        return "0"
    if abs(v) >= 10000:
        return f"{v/10000:.1f}万"
    if abs(v) >= 1000:
        return f"{v/1000:.1f}千"
    return f"{v:.0f}"

def pct(part, total):
    return 0 if total == 0 else part / total * 100

# 行动优先级分：先在 filtered 上生成，再派生 focus_df，避免筛选后重点交易表缺少该字段。
def add_action_priority_score(df):
    df = df.copy()
    if df.empty:
        df["action_priority_score"] = pd.Series(dtype="float")
        return df
    max_amt = max(float(df["net_amount"].max()), 1)
    df["action_priority_score"] = (
        df["rule_hit_count"] * 22
        + df.get("rule_budget_out_risk", 0) * 18
        + df.get("rule_large_amount", 0) * 18
        + df.get("rule_renewal_management", 0) * 16
        + df.get("rule_small_high_freq", 0) * 12
        + df["net_amount"] / max_amt * 14
    ).round(1)
    return df

filtered = add_action_priority_score(filtered)

# 重点关注交易口径：与 SQL 风险识别和作品集保持一致，只统计 risk_level = “重点关注”的 45 笔高优先级交易。
# 规则命中总数仍可在规则排行中查看，二者不是同一个口径。
focus_df = filtered[filtered["risk_level"].eq("重点关注")].copy()

spend_total = filtered["net_amount"].sum()
tx_count = len(filtered)
focus_count = len(focus_df)
rule_hit_total = int((filtered["rule_hit_count"] > 0).sum()) if "rule_hit_count" in filtered.columns else 0
budget_out_amt = filtered.loc[filtered["budget_type"].eq("预算外"), "net_amount"].sum()
budget_out_count = int(filtered["budget_type"].eq("预算外").sum())
high_freq_count = int(filtered["rule_small_high_freq"].sum()) if "rule_small_high_freq" in filtered else 0
renewal_count = int(filtered["rule_renewal_management"].sum()) if "rule_renewal_management" in filtered else 0
large_count = int(filtered["rule_large_amount"].sum()) if "rule_large_amount" in filtered else 0

cat_sum = filtered.groupby("category", as_index=False)["net_amount"].sum().sort_values("net_amount", ascending=False)
scene_sum = filtered.groupby("scene_l2", as_index=False).agg(
    spend=("net_amount", "sum"),
    count=("transaction_id", "count"),
    rule_hit_tx=("rule_hit_count", lambda s: int((s > 0).sum())),
    focus=("risk_level", lambda s: int(s.eq("重点关注").sum())),
).sort_values("spend", ascending=False)
rule_summary = pd.DataFrame([
    {"规则": label, "命中笔数": int(filtered[col].sum()) if col in filtered else 0, "相关金额": filtered.loc[filtered[col].eq(1), "net_amount"].sum() if col in filtered else 0}
    for col, label in RULES.items()
]).sort_values(["命中笔数", "相关金额"], ascending=False)

focus_scene = focus_df.groupby("scene_l2", as_index=False).agg(重点关注笔数=("transaction_id","count"), 重点关注金额=("net_amount","sum")).sort_values("重点关注笔数", ascending=False)

top_cat = cat_sum.iloc[0]["category"] if not cat_sum.empty else "暂无"
top_cat_amt = cat_sum.iloc[0]["net_amount"] if not cat_sum.empty else 0
top_rule = rule_summary.iloc[0]["规则"] if not rule_summary.empty and rule_summary.iloc[0]["命中笔数"] > 0 else "暂无明显规则"
top_focus_scene = focus_scene.iloc[0]["scene_l2"] if not focus_scene.empty else "暂无"

# ---------------- Header ----------------
st.markdown(f"""
<div class="hero">
  <h1>个人收支流水经营分析与风险管控决策看板</h1>
  <div class="subtitle">支出结构 × 预算偏差 × 风险规则识别 × 落地行动清单｜脱敏数据作品集 v02</div>
  <div class="judgement">
    <b>关键判断：</b> 当前筛选范围内，日常经营支出主要集中在 <b>{top_cat}</b>，重点关注交易 <b>{focus_count}</b> 笔，规则命中交易共 <b>{rule_hit_total}</b> 笔；
    风险识别中 <b>{top_rule}</b> 最突出，重点问题场景集中在 <b>{top_focus_scene}</b>。
    建议以“预算外复盘、高频小额控制、续费日历、大额支出冷静期”形成个人经营支出治理闭环。
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("查看动态结论文本"):
    st.write(
        f"在当前筛选条件下，共纳入 {tx_count} 笔日常经营支出，合计 {spend_total:.2f} 元。"
        f"其中预算外支出 {budget_out_amt:.2f} 元，占比 {pct(budget_out_amt, spend_total):.1f}%；"
        f"规则命中交易共 {rule_hit_total} 笔，其中重点关注交易 {focus_count} 笔。"
        f"从规则命中看，{top_rule} 是当前最需要优先处理的规则类型；从场景看，{top_focus_scene} 是重点关注交易最集中的消费场景。"
    )

# Action cards
cols = st.columns(4)
card_data = [
    ("支出结构", f"重点支出：{top_cat}", f"当前支出最高类别为 {top_cat}，金额约 {top_cat_amt:.2f} 元。", "落地动作：按类别设置月度预算上限，优先复盘 Top 支出场景。"),
    ("预算管理", "预算外支出复盘", f"预算外金额约 {budget_out_amt:.2f} 元，占比 {pct(budget_out_amt, spend_total):.1f}%。", "落地动作：建立预算外原因标签，月底复盘是否可提前规划。"),
    ("规则识别", f"突出规则：{top_rule}", f"当前最突出的规则是 {top_rule}，用于定位可优化支出。", "落地动作：把规则阈值固化到提醒工具，形成自动预警。"),
    ("行动优先级", f"重点场景：{top_focus_scene}", f"重点关注交易主要集中在 {top_focus_scene}。", "落地动作：优先处理该场景下的频次、预算和必要性问题。"),
]
for c, item in zip(cols, card_data):
    tag, title, body, action = item
    c.markdown(f"""
    <div class="action-card">
      <span class="tag">{tag}</span>
      <h3>{title}</h3>
      <p>{body}</p>
      <div class="action-note"><b>{action.split('：')[0]}：</b>{action.split('：',1)[1]}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="solution-strip">
  <b>本轮落地解法｜从“记账复盘”到“支出治理机制”：</b>
  预算侧：建立预算内/预算外复盘口径；频次侧：设置高频小额阈值提醒；
  续费侧：建立续费日历与取消评估；大额侧：设置单笔大额冷静期和必要性复核。
</div>
""", unsafe_allow_html=True)

# KPI cards
st.markdown('<div class="section-title">经营健康概览｜先看规模，再找风险摩擦点</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">本模块先判断支出规模、预算压力和规则命中情况，再进入具体场景诊断。</div>', unsafe_allow_html=True)
metric_cols = st.columns(5)
metrics = [
    ("日常经营支出", f"{spend_total:.0f} 元", f"{tx_count} 笔交易"),
    ("重点关注交易", f"{focus_count} 笔", f"规则命中共 {rule_hit_total} 笔"),
    ("预算外支出", f"{budget_out_amt:.0f} 元", f"{budget_out_count} 笔"),
    ("高频小额命中", f"{high_freq_count} 笔", "隐性累计成本"),
    ("续费/大额命中", f"{renewal_count + large_count} 笔", "需专项复核"),
]
for col, (label, value, hint) in zip(metric_cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="hint">{hint}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Section 1 ----------------
st.markdown('<div class="section-title">一、支出结构与预算偏差诊断</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">回答：钱主要花在哪里？哪些支出超出预算或存在结构性优化空间？</div>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("当前筛选条件下没有数据，请调整筛选器。")
    st.stop()

c1, c2 = st.columns([1,1])
with c1:
    fig_cat = px.bar(cat_sum.head(10).sort_values("net_amount"), x="net_amount", y="category", orientation="h", text="net_amount", title="类别支出排行")
    fig_cat.update_traces(texttemplate="%{text:.0f}元", marker_color="#2563eb")
    fig_cat.update_layout(height=380, margin=dict(l=10,r=20,t=55,b=20), xaxis_title="支出金额（元）", yaxis_title="类别")
    st.plotly_chart(fig_cat, use_container_width=True)
with c2:
    daily = filtered.groupby("transaction_date", as_index=False)["net_amount"].sum().sort_values("transaction_date")
    fig_daily = px.line(daily, x="transaction_date", y="net_amount", markers=True, title="每日支出趋势")
    fig_daily.update_traces(line_color="#2563eb")
    fig_daily.update_layout(height=380, margin=dict(l=10,r=20,t=55,b=20), xaxis_title="日期", yaxis_title="支出金额（元）")
    st.plotly_chart(fig_daily, use_container_width=True)

c3, c4 = st.columns([1,1])
with c3:
    budget = filtered.groupby("budget_type", as_index=False)["net_amount"].sum().sort_values("net_amount", ascending=False)
    fig_budget = px.pie(budget, names="budget_type", values="net_amount", hole=.48, title="预算内外结构")
    fig_budget.update_layout(height=340, margin=dict(l=10,r=20,t=55,b=20))
    st.plotly_chart(fig_budget, use_container_width=True)
with c4:
    necessity = filtered.groupby("necessity_type", as_index=False)["net_amount"].sum().sort_values("net_amount", ascending=False)
    fig_nec = px.bar(necessity, x="necessity_type", y="net_amount", text="net_amount", title="必要性结构")
    fig_nec.update_traces(texttemplate="%{text:.0f}元", marker_color="#60a5fa")
    fig_nec.update_layout(height=340, margin=dict(l=10,r=20,t=55,b=20), xaxis_title="必要性", yaxis_title="金额（元）")
    st.plotly_chart(fig_nec, use_container_width=True)

# ---------------- Section 2 ----------------
st.markdown('<div class="section-title">二、风险规则命中与问题定位</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">回答：哪些规则命中最多？哪些消费场景属于高金额、高频次或高规则命中区域？</div>', unsafe_allow_html=True)

c5, c6 = st.columns([1,1])
with c5:
    rs = rule_summary.copy()
    fig_rule = px.bar(rs.sort_values("命中笔数"), x="命中笔数", y="规则", orientation="h", text="命中笔数", title="风险规则命中排行")
    fig_rule.update_traces(marker_color="#ef4444", textposition="outside")
    fig_rule.update_layout(height=390, margin=dict(l=10,r=45,t=55,b=20), xaxis_title="命中笔数", yaxis_title="规则")
    st.plotly_chart(fig_rule, use_container_width=True)
with c6:
    scene_diag = scene_sum.copy()
    scene_diag["预算外金额"] = filtered[filtered["budget_type"].eq("预算外")].groupby("scene_l2")["net_amount"].sum().reindex(scene_diag["scene_l2"]).fillna(0).values
    fig_scene = px.scatter(scene_diag, x="spend", y="focus", size="count", color="预算外金额", hover_name="scene_l2", title="场景金额 × 重点关注矩阵", color_continuous_scale="OrRd")
    if not scene_diag.empty:
        fig_scene.add_vline(x=scene_diag["spend"].mean(), line_dash="dash", line_color="gray", annotation_text="平均金额")
        fig_scene.add_hline(y=scene_diag["focus"].mean(), line_dash="dash", line_color="gray", annotation_text="平均关注")
    fig_scene.update_traces(marker=dict(sizemin=8, opacity=.78))
    fig_scene.update_layout(height=390, margin=dict(l=10,r=20,t=55,b=20), xaxis_title="场景支出金额（元）", yaxis_title="重点关注笔数")
    st.plotly_chart(fig_scene, use_container_width=True)

# Weekday heatmap-ish bar
weekday_order = ["一","二","三","四","五","六","日"]
weekday = filtered.groupby("weekday_name", as_index=False).agg(金额=("net_amount","sum"), 笔数=("transaction_id","count"))
weekday["weekday_name"] = pd.Categorical(weekday["weekday_name"], categories=weekday_order, ordered=True)
weekday = weekday.sort_values("weekday_name")
fig_week = px.bar(weekday, x="weekday_name", y="金额", text="金额", title="星期维度支出观察")
fig_week.update_traces(texttemplate="%{text:.0f}元", marker_color="#93c5fd")
fig_week.update_layout(height=300, margin=dict(l=10,r=20,t=55,b=20), xaxis_title="星期", yaxis_title="支出金额（元）")
st.plotly_chart(fig_week, use_container_width=True)

# ---------------- Section 3 ----------------
st.markdown('<div class="section-title">三、重点交易跟进池与行动建议</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">回答：哪些交易需要优先复盘？每笔交易为什么被关注？下一步该如何处理？</div>', unsafe_allow_html=True)

focus_display = focus_df.copy()
if "action_priority_score" not in focus_display.columns:
    focus_display = add_action_priority_score(focus_display)
focus_display = focus_display.sort_values("action_priority_score", ascending=False)
if focus_display.empty:
    st.info("当前筛选范围暂无重点关注交易。")
else:
    left, right = st.columns([1.35, .9])
    with left:
        fig_tx = px.scatter(
            focus_display,
            x="net_amount",
            y="rule_hit_count",
            color="risk_tag",
            size="net_amount",
            hover_name="transaction_id",
            hover_data=["transaction_date", "category", "scene_l2", "budget_type", "hit_rule_names", "risk_reason", "action_suggestion"],
            title="重点交易散点图：金额 × 规则命中数"
        )
        fig_tx.add_vline(x=focus_display["net_amount"].mean(), line_dash="dash", line_color="gray", annotation_text="平均金额")
        fig_tx.add_hline(y=focus_display["rule_hit_count"].mean(), line_dash="dash", line_color="gray", annotation_text="平均命中")
        fig_tx.update_layout(height=460, margin=dict(l=10,r=20,t=55,b=20), xaxis_title="单笔金额（元）", yaxis_title="命中规则数")
        st.plotly_chart(fig_tx, use_container_width=True)
    with right:
        tx_options = focus_display["transaction_id"].astype(str).tolist()
        tx_choice = st.selectbox("选择重点交易查看处理建议", tx_options)
        row = focus_display[focus_display["transaction_id"].astype(str).eq(tx_choice)].iloc[0]
        st.markdown("### 交易行动建议")
        st.write(f"**交易 ID：** {row['transaction_id']}｜**日期：** {row['transaction_date'].date() if pd.notna(row['transaction_date']) else ''}")
        st.write(f"**类别/场景：** {row['category']} / {row['scene_l2']}")
        st.write(f"**金额：** {row['net_amount']:.2f} 元｜**预算类型：** {row['budget_type']}｜**风险标签：** {row['risk_tag']}")
        st.write(f"**命中规则：** {row['hit_rule_names']}")
        st.markdown(f"<div class='recommend-box'><b>风险原因：</b>{row['risk_reason']}<br><br><b>建议动作：</b>{row['action_suggestion']}</div>", unsafe_allow_html=True)
        st.write(f"**行动优先级分：** {row['action_priority_score']}")

# ---------------- Section 4 ----------------
st.markdown('<div class="section-title">四、支出治理机制与标准化方案</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">回答：哪些问题不能只靠人工月底复盘，而应沉淀成固定规则、提醒机制和复盘动作？</div>', unsafe_allow_html=True)

solution_rows = []
def add_solution(problem, condition_col, lead, team, sla, solution):
    if condition_col in filtered.columns:
        d = filtered[filtered[condition_col].eq(1)].copy()
    else:
        d = filtered.iloc[0:0].copy()
    solution_rows.append({
        "问题类型": problem,
        "命中笔数": len(d),
        "相关金额": d["net_amount"].sum() if not d.empty else 0,
        "牵头角色": lead,
        "协同角色": team,
        "建议频率/SLA": sla,
        "标准化处理方案": solution,
    })
solution_defs = [
    ("高频小额", "rule_small_high_freq", "本人", "预算复盘/工具提醒", "每周复盘", "设置同场景频次阈值，超过阈值自动提醒；月底检查是否属于隐性累计成本。"),
    ("预算外风险", "rule_budget_out_risk", "本人", "预算管理", "月度复盘", "建立预算外原因标签，区分临时必要支出和可提前规划支出，形成下月预算修正。"),
    ("续费管理", "rule_renewal_management", "本人", "订阅清单", "续费前3天", "建立续费日历，续费前判断是否仍使用、是否可替代、是否需要取消。"),
    ("单笔大额", "rule_large_amount", "本人", "消费审批", "消费前/当天", "设置单笔大额冷静期，记录购买理由、必要性和预算来源。"),
    ("外卖频次", "rule_food_delivery", "本人", "餐饮计划", "每周复盘", "设置每周外卖频次上限，优先用备餐、食堂或到店场景替代。"),
]
for item in solution_defs:
    add_solution(*item)
solution_table = pd.DataFrame(solution_rows)
solution_table["治理优先级分"] = (
    solution_table["命中笔数"] * 3 +
    solution_table["相关金额"] / max(solution_table["相关金额"].max(), 1) * 30
).round(1)
solution_table = solution_table.sort_values("治理优先级分", ascending=False)

c7, c8 = st.columns([1,1])
with c7:
    fig_sol = px.bar(solution_table.sort_values("治理优先级分"), x="治理优先级分", y="问题类型", orientation="h", text="治理优先级分", title="治理机制优先级排行")
    fig_sol.update_traces(marker_color="#2563eb", textposition="outside")
    fig_sol.update_layout(height=400, margin=dict(l=10,r=55,t=55,b=20), xaxis_title="治理优先级分", yaxis_title="问题类型")
    st.plotly_chart(fig_sol, use_container_width=True)
with c8:
    st.markdown("### Top 治理问题与落地方案")
    show_sol = solution_table[["问题类型", "治理优先级分", "命中笔数", "相关金额", "牵头角色", "协同角色", "建议频率/SLA", "标准化处理方案"]].copy()
    show_sol["相关金额"] = show_sol["相关金额"].map(lambda x: f"{x:.1f} 元")
    st.dataframe(show_sol, use_container_width=True, hide_index=True)

st.markdown("""
<div class="solution-strip">
  <b>三层处理机制：</b>
  L1 高频小额/外卖频次：用阈值提醒直接处理；
  L2 预算外/单笔大额：进入月度预算复盘和必要性复核；
  L3 续费管理：进入订阅清单、续费日前提醒和取消评估机制。
</div>
""", unsafe_allow_html=True)

# ---------------- Section 5 ----------------
st.markdown('<div class="section-title">五、可执行行动清单与下载</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">把图表结论转成可执行表：优先处理什么、为什么处理、怎么处理、多久复盘。</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["重点交易跟进清单", "治理机制处理清单", "预算外复盘清单"])
with tab1:
    follow_cols = ["transaction_id", "transaction_date", "category", "scene_l1", "scene_l2", "net_amount", "budget_type", "risk_tag", "hit_rule_names", "rule_hit_count", "action_priority_score", "risk_reason", "action_suggestion"]
    follow = focus_display[follow_cols].copy() if not focus_display.empty else pd.DataFrame(columns=follow_cols)
    st.dataframe(follow, use_container_width=True, hide_index=True, height=360)
    st.download_button("下载重点交易跟进清单 CSV", follow.to_csv(index=False).encode("utf-8-sig"), "cashflow_focus_action_list.csv", "text/csv")
with tab2:
    st.dataframe(solution_table, use_container_width=True, hide_index=True, height=330)
    st.download_button("下载治理机制处理清单 CSV", solution_table.to_csv(index=False).encode("utf-8-sig"), "cashflow_governance_solution_list.csv", "text/csv")
with tab3:
    budget_out = filtered[filtered["budget_type"].eq("预算外")].copy().sort_values("net_amount", ascending=False)
    budget_cols = ["transaction_id", "transaction_date", "category", "scene_l2", "net_amount", "risk_tag", "hit_rule_names", "risk_reason", "action_suggestion"]
    st.dataframe(budget_out[budget_cols], use_container_width=True, hide_index=True, height=330)
    st.download_button("下载预算外复盘清单 CSV", budget_out[budget_cols].to_csv(index=False).encode("utf-8-sig"), "cashflow_budget_out_review_list.csv", "text/csv")

st.markdown("<div class='footer-note'>说明：本看板基于脱敏个人收支流水构建，用于展示经营分析、预算复盘、规则识别和自动化行动清单能力；不代表企业真实数据。</div>", unsafe_allow_html=True)
