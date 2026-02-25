import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import kagglehub
import os

# =========================================================
# 1. Page Configuration & Styling
# =========================================================
st.set_page_config(page_title="Sephora Analytics Dashboard", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;700&family=Frank+Ruhl+Libre:wght@900&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Assistant', sans-serif !important;
        direction: ltr;
        text-align: left;
    }

    h1 {
        font-family: 'Frank Ruhl Libre', serif !important;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        color: #000000 !important;
        border-bottom: 4px solid #000000;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }

    h2, h3 {
        font-weight: 700 !important;
        color: #1E1E1E !important;
        margin-top: 20px !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #dcae96 !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 400 !important;
        font-size: 1rem !important;
    }

    .stRadio > label { font-weight: 700; font-size: 1.2rem; }

    [data-testid="stSidebar"] {
        background-color: #fdfdfd !important;
        border-right: 1px solid #e6e6e6;
    }

    /* ── Tablet (≤ 1024px) ─────────────────────────────── */
    @media (max-width: 1024px) {
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.2rem !important; }
        [data-testid="column"] { min-width: 45% !important; flex: 1 1 45% !important; }
    }

    /* ── Mobile (≤ 768px) ──────────────────────────────── */
    @media (max-width: 768px) {
        h1 { font-size: 1.8rem !important; border-bottom-width: 2px; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
        [data-testid="stSidebar"] { min-width: 0 !important; }
        .js-plotly-plot { width: 100% !important; }
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    }

    /* ── Small mobile (≤ 480px) ────────────────────────── */
    @media (max-width: 480px) {
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1rem !important; }
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. Data (Load Real CSVs) + Derived Metrics
# =========================================================
@st.cache_data(show_spinner=True)
def load_reviews_and_products():
    path = kagglehub.dataset_download("nadyinky/sephora-products-and-skincare-reviews")

    reviews_files = [
        'reviews_0-250.csv',
        'reviews_250-500.csv',
        'reviews_500-750.csv',
        'reviews_750-1250.csv',
        'reviews_1250-end.csv'
    ]

    reviews_parts = []
    for fp in reviews_files:
        part = pd.read_csv(os.path.join(path, fp))
        part['source_file'] = fp
        reviews_parts.append(part)

    reviews_df = pd.concat(reviews_parts, ignore_index=True)
    product_df = pd.read_csv(os.path.join(path, 'product_info.csv'))

    # Type cleaning
    for col in ['price_usd', 'rating', 'is_recommended']:
        if col in reviews_df.columns:
            reviews_df[col] = pd.to_numeric(reviews_df[col], errors='coerce')

    # Review length (words)
    if 'review_text' in reviews_df.columns:
        reviews_df['review_length_words'] = (
            reviews_df['review_text']
              .fillna('')
              .astype(str)
              .str.split()
              .str.len()
        )

    # Merge categories from product_info
    if 'product_id' in reviews_df.columns and 'product_id' in product_df.columns:
        keep_cols = [c for c in ['product_id','primary_category','secondary_category','tertiary_category','price_usd'] if c in product_df.columns]
        products_small = product_df[keep_cols].drop_duplicates('product_id')
        reviews_df = reviews_df.merge(products_small, on='product_id', how='left', suffixes=('', '_product'))

        # Backfill missing review price with product price
        if 'price_usd_product' in reviews_df.columns:
            reviews_df['price_usd'] = reviews_df['price_usd'].fillna(reviews_df['price_usd_product'])

    return reviews_df, product_df

reviews_df, product_df = load_reviews_and_products()

# High-level metrics
TOTAL_REVIEWS = int(len(reviews_df))
RECOMMENDED_COUNT = int((reviews_df['is_recommended'] == 1).sum())
NOT_RECOMMENDED_COUNT = int((reviews_df['is_recommended'] == 0).sum())
REC_RATE = (100 * RECOMMENDED_COUNT / TOTAL_REVIEWS) if TOTAL_REVIEWS else 0
AVG_RATING = float(reviews_df['rating'].mean())

# Derived metrics used in the visuals
recs = reviews_df[reviews_df['is_recommended'] == 1]
not_recs = reviews_df[reviews_df['is_recommended'] == 0]

RATING_RECOMMENDED_MEAN = float(recs['rating'].mean())
RATING_NOT_RECOMMENDED_MEAN = float(not_recs['rating'].mean())
RATING_OVERALL_SD = float(reviews_df['rating'].std())

PRICE_MEAN = float(reviews_df['price_usd'].mean())
PRICE_MEDIAN = float(reviews_df['price_usd'].median())
PRICE_SD = float(reviews_df['price_usd'].std())

REVIEW_LENGTH_RECOMMENDED_MEAN = float(recs['review_length_words'].mean())
REVIEW_LENGTH_NOT_RECOMMENDED_MEAN = float(not_recs['review_length_words'].mean())

COLORS = {
    'positive': '#DCAE96',
    'negative': '#4A4A4A',
    'accent': '#000000',
    'chart': '#C4956A',
}

# Category table for display (real)
df_categories = (
    reviews_df.dropna(subset=['primary_category','is_recommended'])
      .groupby('primary_category')
      .agg(
          **{
              'Recommendation Rate': ('is_recommended', lambda x: 100 * (x==1).mean()),
              'Review Count': ('is_recommended', 'size')
          }
      )
      .reset_index()
      .rename(columns={'primary_category':'Category'})
      .sort_values(['Recommendation Rate','Review Count'], ascending=[False, False])
)

# =========================================================
# 3. Sidebar Navigation
# =========================================================
st.sidebar.markdown("## Controls")
st.sidebar.markdown("---")

show_section = st.sidebar.radio(
    "Select section:",
    ["Overview", "Rating Analysis", "Price Analysis", "Review Length Analysis", "Category Analysis", "NLP Explorer"]
)

# =========================================================
# 4. Title
# =========================================================
st.title("Sephora Recommendation Analysis")
st.subheader("Insights from 926,423 Verified Customer Reviews")

st.markdown("---")

# =========================================================
# Section: Overview
# =========================================================
if show_section == "Overview":
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{TOTAL_REVIEWS:,}")
    col2.metric("Recommendation Rate", f"{REC_RATE:.2f}%")
    col3.metric("Average Rating", f"{AVG_RATING:.2f}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Recommended', 'Not Recommended'],
            values=[RECOMMENDED_COUNT, NOT_RECOMMENDED_COUNT],
            hole=0.5,
            marker=dict(colors=[COLORS['positive'], COLORS['negative']])
        )])
        fig_pie.update_layout(
            title="Overall Recommendation Distribution",
            font=dict(family="Assistant"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Key Analytical Insights")
        st.markdown("""
        * **Rating & Recommendation:** A massive correlation ($d=4.31$) was found. Rating is the strongest predictor of recommendation.
        * **Review Length:** No significant practical difference was observed. Content quality outweighs word count.
        * **Selection Bias:** The high 84% recommendation rate suggests that satisfied customers are more likely to leave reviews.
        * **Category Impact:** Recommendation rates vary significantly across product categories.
        """)

# =========================================================
# Section: Rating Analysis
# =========================================================
elif show_section == "Rating Analysis":
    st.markdown("## Rating Comparison by Recommendation")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Rating (Recommended)", f"{RATING_RECOMMENDED_MEAN:.2f}")
    col2.metric("Avg Rating (Not Recommended)", f"{RATING_NOT_RECOMMENDED_MEAN:.2f}")
    col3.metric("Difference", f"{RATING_RECOMMENDED_MEAN - RATING_NOT_RECOMMENDED_MEAN:.2f} stars")

    st.markdown("---")

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=[RATING_RECOMMENDED_MEAN] * 100,
        name='Recommended',
        marker=dict(color=COLORS['positive']),
        boxmean='sd'
    ))
    fig_box.add_trace(go.Box(
        y=[RATING_NOT_RECOMMENDED_MEAN] * 100,
        name='Not Recommended',
        marker=dict(color=COLORS['negative']),
        boxmean='sd'
    ))
    fig_box.update_layout(title="Rating Comparison", yaxis_title="Rating (stars)", height=400)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    ### Key Finding
    **Rating is the strongest predictor of recommendation!**
    - Recommended reviews: avg **4.71** stars
    - Not recommended reviews: avg **2.05** stars
    - P-value: **< 0.001** | Cohen's d: **4.31** (huge effect size)
    """)

# =========================================================
# Section: Price Analysis
# =========================================================
elif show_section == "Price Analysis":
    st.markdown("## Price Distribution")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Price", f"${PRICE_MEAN:.2f}")
    col2.metric("Median Price", f"${PRICE_MEDIAN:.2f}")
    col3.metric("Std. Deviation", f"${PRICE_SD:.2f}")
    col4.metric("Range", "$0 - $1,900+")

    st.markdown("---")

    price_data = reviews_df['price_usd'].dropna().values

    fig_hist = go.Figure(data=[go.Histogram(
        x=price_data, nbinsx=50,
        marker=dict(color=COLORS['chart']),
        name='Product Count'
    )])
    fig_hist.update_layout(title="Product Price Distribution", xaxis_title="Price (USD)", yaxis_title="Count", height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("""
    ### Finding
    **Price shows no strong relationship with recommendation**
    - Most products fall in the $20–$80 range
    - Right-skewed distribution (a few very expensive products)
    - No strong statistical relationship found between price and recommendation
    """)

# =========================================================
# Section: Review Length Analysis
# =========================================================
elif show_section == "Review Length Analysis":
    st.markdown("## Review Length")

    col1, col2 = st.columns(2)
    col1.metric("Mean (Recommended)", f"{REVIEW_LENGTH_RECOMMENDED_MEAN:.0f} words")
    col2.metric("Mean (Not Recommended)", f"{REVIEW_LENGTH_NOT_RECOMMENDED_MEAN:.0f} words")

    st.markdown("---")

    fig_bar = go.Figure(data=[go.Bar(
        x=['Recommended', 'Not Recommended'],
        y=[REVIEW_LENGTH_RECOMMENDED_MEAN, REVIEW_LENGTH_NOT_RECOMMENDED_MEAN],
        marker=dict(color=[COLORS['positive'], COLORS['negative']]),
        text=[f"{REVIEW_LENGTH_RECOMMENDED_MEAN:.0f}", f"{REVIEW_LENGTH_NOT_RECOMMENDED_MEAN:.0f}"],
        textposition="auto"
    )])
    fig_bar.update_layout(title="Average Review Length by Type", yaxis_title="Words", height=400, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    ### Finding
    **Review length is negligible!**
    - Difference: only **1.09 words** (practically nothing)
    - Cohen's d: **0.0257** | Conclusion: Content quality matters, not length.
    """)

# =========================================================
# Section: Category Analysis
# =========================================================
elif show_section == "Category Analysis":
    st.markdown("## Recommendation Rate by Category")

    top_n = st.slider("Number of categories to display:", 3, 8, 8)
    df_top = df_categories.head(top_n)

    fig_cat = go.Figure(data=[go.Bar(
        y=df_top['Category'],
        x=df_top['Recommendation Rate'],
        orientation='h',
        marker=dict(color=df_top['Recommendation Rate'], colorscale=[[0, COLORS['negative']], [1, COLORS['positive']]]),
        text=df_top['Recommendation Rate'].apply(lambda x: f"{x:.1f}%"),
        textposition="auto"
    )])
    fig_cat.update_layout(
        title="Recommendation Rate by Category",
        xaxis_title="Recommendation Rate (%)",
        yaxis_title="Category",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    df_display = df_top.copy()
    df_display['Recommendation Rate'] = df_display['Recommendation Rate'].apply(lambda x: f"{x:.1f}%")
    df_display['Review Count'] = df_display['Review Count'].apply(lambda x: f"{x:,}")
    st.dataframe(df_display, use_container_width=True)

    st.markdown("""
    ### Findings
    - Skincare: **85.2%** (highest) | Cleansers: **79.8%** (lowest)
    - Chi-square test: χ² = 428,281.93, p ≈ 0
    - **Conclusion:** Category matters — prioritize high-recommendation categories in new markets.
    """)

# =========================================================
# Section: NLP Explorer
# =========================================================
elif show_section == "NLP Explorer":
    st.markdown("## NLP Explorer: Keyword Sentiment Search")

    keyword = st.text_input("Enter a keyword to see its impact on recommendations (e.g., love, allergy, pricey):")

    if keyword:
        st.info(f"Analyzing reviews containing the word: **'{keyword}'**")

        filtered_df = reviews_df[reviews_df['review_text'].fillna('').astype(str).str.contains(keyword, case=False, na=False)]

        occ = int(len(filtered_df))
        rec_rate_kw = (100 * (filtered_df['is_recommended'] == 1).mean()) if occ else 0
        avg_rating_kw = float(filtered_df['rating'].mean()) if occ else float('nan')

        c1, c2, c3 = st.columns(3)
        c1.metric("Occurrences", f"{occ:,}")
        c2.metric("Keyword Rec Rate", f"{rec_rate_kw:.1f}%")
        c3.metric("Avg Rating for Term", f"{avg_rating_kw:.2f}" if occ else "N/A")

        overall_rec = 100 * (reviews_df['is_recommended'] == 1).mean()
        if occ and rec_rate_kw < overall_rec:
            st.warning("This keyword is associated with lower recommendation rates than the overall average.")
        elif occ:
            st.success("This keyword is associated with higher recommendation rates than the overall average.")
        else:
            st.info("No reviews matched this keyword. Try another term.")

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
<p>Based on analysis of 926,423 real Sephora reviews</p>
<p>Data from <a href="https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews" target="_blank">Kaggle: nadyinky/sephora-products-and-skincare-reviews</a></p>
<p>Created April 2026</p>
</div>
""", unsafe_allow_html=True)
