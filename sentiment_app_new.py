

import streamlit as st
import pickle
import re
import numpy as np
import plotly.graph_objects as go
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
    .main { background-color: #0f1117; }
    .block-container { padding-top: 3.5rem; }
    .title-text {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle-text {
        color: #64748b; font-size: 1rem;
        margin-top: 0.2rem; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e2330; border: 1px solid #2d3748;
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-label { color: #64748b; font-size: 0.8rem;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.4rem; }
    .metric-value { color: #e2e8f0; font-size: 1.5rem; font-weight: 700; }
    .result-positive {
        background: linear-gradient(135deg, #1a2e1a, #1a3a1a);
        border: 2px solid #4ade80; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin: 1rem 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #2e1a1a, #3d1f1f);
        border: 2px solid #f87171; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin: 1rem 0;
    }
    .result-label { font-size: 0.8rem; letter-spacing: 2px;
        text-transform: uppercase; margin-bottom: 0.3rem; }
    .result-emoji { font-size: 2.8rem; margin: 0.3rem 0; }
    .result-value { color: white; font-size: 2rem; font-weight: 800;
        font-family: 'Syne', sans-serif; }
    .result-conf { color: #64748b; font-size: 0.85rem; margin-top: 0.3rem; }
    .word-tag-pos {
        display: inline-block; background: rgba(74,222,128,0.15);
        border: 1px solid rgba(74,222,128,0.4); color: #4ade80;
        padding: 2px 8px; border-radius: 20px; margin: 2px; font-size: 0.8rem;
    }
    .word-tag-neg {
        display: inline-block; background: rgba(248,113,113,0.15);
        border: 1px solid rgba(248,113,113,0.4); color: #f87171;
        padding: 2px 8px; border-radius: 20px; margin: 2px; font-size: 0.8rem;
    }
    .word-tag-neu {
        display: inline-block; background: rgba(100,116,139,0.15);
        border: 1px solid rgba(100,116,139,0.4); color: #94a3b8;
        padding: 2px 8px; border-radius: 20px; margin: 2px; font-size: 0.8rem;
    }
    .result-mixed {
        background: linear-gradient(135deg, #2a2a1a, #1e2a2a);
        border: 2px solid #facc15; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin: 1rem 0;
    }
    .placeholder-box {
        background: #1e2330; border: 1px dashed #2d3748;
        border-radius: 12px; padding: 2.5rem;
        text-align: center; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('sentiment_model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, vectorizer, info

model, vectorizer, model_info = load_model()
STOPWORDS = set(stopwords.words('english'))

POSITIVE_WORDS = {
    'excellent', 'great', 'amazing', 'perfect', 'wonderful',
    'brilliant', 'fantastic', 'outstanding', 'superb', 'best',
    'love', 'loved', 'beautiful', 'entertaining', 'fun',
    'enjoyable', 'impressive', 'masterpiece', 'extraordinary',
    'stunning', 'breathtaking', 'recommend', 'superb', 'delightful',
    'captivating', 'riveting', 'engaging', 'magnificent', 'spectacular',
    'gem', 'clever', 'heartfelt', 'touching', 'moving', 'hilarious',
    'thrilling', 'gripping', 'solid', 'polished', 'refreshing'
}
NEGATIVE_WORDS = {
    'worst', 'awful', 'bad', 'terrible', 'boring', 'waste',
    'poor', 'fails', 'horrible', 'disappointing', 'dull',
    'stupid', 'pathetic', 'ridiculous', 'avoid', 'nothing',
    'disappointment', 'poorly', 'painfully', 'unbearable',
    'rushed', 'predictable', 'forgettable', 'generic', 'mediocre',
    'bland', 'flat', 'shallow', 'incoherent', 'confusing',
    'messy', 'weak', 'forced', 'unconvincing', 'overlong',
    'tedious', 'lifeless', 'pointless', 'hollow', 'failed',
    'asleep', 'special', 'sense', 'place', 'whatsoever'
}


MIXED_THRESHOLD = 0.65


def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)


def get_positive_label():
    """Determine which label value corresponds to 'positive' sentiment."""
    classes = model.classes_
   
    if hasattr(model_info, 'get') and 'positive_label' in model_info:
        return model_info['positive_label']
 
    if len(classes) == 2:
        
        if 'positive' in [str(c).lower() for c in classes]:
            for c in classes:
                if str(c).lower() == 'positive':
                    return c
        
        return max(classes)
    return 


POSITIVE_LABEL = get_positive_label()


def is_positive(pred):
    """Check if a prediction is positive, handling int/str/numpy types."""
    return int(pred) == int(POSITIVE_LABEL) or str(pred).lower() == 'positive'


def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    if hasattr(model, 'decision_function'):
        score = model.decision_function(vec)[0]
        confidence = 1 / (1 + np.exp(-abs(score)))
    else:
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)
    return pred, confidence, cleaned


def highlight_words(text):
    words = text.split()
    html_parts = []
    for word in words:
        clean_word = word.lower().strip('.,!?";:')
        if clean_word in POSITIVE_WORDS:
            html_parts.append(f'<span class="word-tag-pos">{word}</span>')
        elif clean_word in NEGATIVE_WORDS:
            html_parts.append(f'<span class="word-tag-neg">{word}</span>')
        else:
            html_parts.append(f'<span class="word-tag-neu">{word}</span>')
    return ' '.join(html_parts)



st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="title-text">🎬 Sentiment Analyzer</p>',
            unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">Model</div>'
                '<div class="metric-value">Linear SVM</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Accuracy</div>'
                '<div class="metric-value">89.8%</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">ROC-AUC</div>'
                '<div class="metric-value">0.963</div></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-label">Training Data</div>'
                '<div class="metric-value">50,000</div></div>',
                unsafe_allow_html=True)

st.markdown("---")


left_input, right_input = st.columns(2, gap="large")

with left_input:
    st.markdown("### ✍️ Single Review")
    col1, col2, col3 = st.columns(3)
    sample_pos = "This film was absolutely brilliant! The performances were extraordinary and the story kept me engaged throughout. A true masterpiece."
    sample_neg = "Terrible waste of time. The plot made no sense, acting was awful, and I nearly fell asleep. Completely disappointing experience."
    sample_mix = "The movie had some great visuals but the story was boring and predictable. Mixed feelings overall about this one."

    if col1.button("😊 Positive", use_container_width=True):
        st.session_state.review_text = sample_pos
        
        st.session_state.single_done = False
    if col2.button("😞 Negative", use_container_width=True):
        st.session_state.review_text = sample_neg
        st.session_state.single_done = False
    if col3.button("😐 Mixed", use_container_width=True):
        st.session_state.review_text = sample_mix
        st.session_state.single_done = False

    review_text = st.text_area(
        "Paste any review or text:",
        value=st.session_state.get('review_text', ''),
        height=150,
        placeholder="Type or paste a review here...",
        key="single_input"
    )
    analyze_btn = st.button("🔍 Analyze Sentiment",
                            type="primary", use_container_width=True)

    
    if analyze_btn and review_text.strip():
        pred, confidence, _ = predict_sentiment(review_text)
        st.session_state.single_pred  = pred
        st.session_state.single_conf  = confidence
        st.session_state.single_text  = review_text
        st.session_state.single_done  = True

with right_input:
    st.markdown("### 📋 Batch Analysis")
    st.markdown("<div style='height:2.35rem'></div>", unsafe_allow_html=True)
    batch_text = st.text_area(
        "Enter reviews (one per line):",
        height=150,
        placeholder="Review 1\nReview 2\nReview 3",
        key="batch_input"
    )
    batch_btn = st.button("📊 Analyze All", use_container_width=True)

    if batch_btn and batch_text.strip():
        reviews = [r.strip() for r in batch_text.split('\n') if r.strip()]
        preds = []
        confs = []
        batch_results = []
        for rev in reviews:
            pred, conf, _ = predict_sentiment(rev)
            preds.append(pred)
            confs.append(conf)
            if conf < MIXED_THRESHOLD:
                label = '😐 Mixed'
            elif is_positive(pred):
                label = '😊 Positive'
            else:
                label = '😞 Negative'
            batch_results.append({
                'Review':     rev[:55] + '...' if len(rev) > 55 else rev,
                'Sentiment':  label,
                'Confidence': f'{conf*100:.1f}%'
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(batch_results),
                     use_container_width=True, hide_index=True)
        st.session_state.batch_preds   = preds
        st.session_state.batch_confs   = confs
        st.session_state.batch_reviews = reviews
        st.session_state.batch_done    = True

st.markdown("---")


left_result, right_result = st.columns(2, gap="large")


with left_result:
    st.markdown("### 📊 Single Review Result")

   
    if st.session_state.get('single_done') and st.session_state.get('single_text'):
        pred        = st.session_state.single_pred
        confidence  = st.session_state.single_conf
        review_text = st.session_state.single_text

        sentiment_positive = is_positive(pred)
        is_mixed = confidence < MIXED_THRESHOLD

        if is_mixed:
            sentiment_label = "MIXED"
            emoji           = "😐"
            box_class       = "result-mixed"
            color           = "#facc15"
        elif sentiment_positive:
            sentiment_label = "POSITIVE"
            emoji           = "😊"
            box_class       = "result-positive"
            color           = "#4ade80"
        else:
            sentiment_label = "NEGATIVE"
            emoji           = "😞"
            box_class       = "result-negative"
            color           = "#f87171"

        conf_note = " · low confidence — mixed signals detected" if is_mixed else ""

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-label" style="color:{color}">Sentiment Result</div>
            <div class="result-emoji">{emoji}</div>
            <div class="result-value">{sentiment_label}</div>
            <div class="result-conf">Confidence: {confidence*100:.1f}%{conf_note}</div>
        </div>
        """, unsafe_allow_html=True)

        gauge_color = color
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={'suffix': '%', 'font': {'color': gauge_color, 'size': 26}},
            gauge={
                'axis': {'range': [50, 100]},
                'bar':  {'color': gauge_color},
                'bgcolor': "#1e2330",
                'bordercolor': "#2d3748",
                'steps': [
                    {'range': [50, 70],  'color': '#1e2330'},
                    {'range': [70, 85],  'color': '#252a3a'},
                    {'range': [85, 100], 'color': '#2a3040'},
                ],
            },
            title={'text': "Model Confidence",
                   'font': {'color': '#94a3b8', 'size': 12}}
        ))
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'},
            height=220, margin=dict(t=50, b=0)
        )
        st.plotly_chart(fig1, use_container_width=True, key="gauge_chart")

        st.markdown("**🔍 Key Words Detected:**")
        st.markdown(f"<div style='line-height:2.2'>{highlight_words(review_text)}</div>",
                    unsafe_allow_html=True)
        st.caption("🟢 Positive  🔴 Negative  ⚫ Neutral")

        words     = review_text.split()
        pos_count = sum(1 for w in words if w.lower().strip('.,!?') in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w.lower().strip('.,!?') in NEGATIVE_WORDS)
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Words",    len(words))
        s2.metric("Positive Words", pos_count)
        s3.metric("Negative Words", neg_count)

    else:
        st.markdown("""
        <div class="placeholder-box">
            <div style='font-size:2.5rem'>🎬</div>
            <div style='color:#64748b; margin-top:0.8rem'>
                Enter a review above and click<br>
                <b style='color:#a78bfa'>Analyze Sentiment</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


with right_result:
    st.markdown("### 📊 Batch Analysis Result")

    if st.session_state.get('batch_done'):
        preds   = st.session_state.batch_preds
        reviews = st.session_state.batch_reviews

        confs     = st.session_state.get('batch_confs', [1.0]*len(preds))
        pos_flags = [is_positive(p) and c >= MIXED_THRESHOLD for p, c in zip(preds, confs)]
        mix_flags = [c < MIXED_THRESHOLD for c in confs]
        neg_flags = [not is_positive(p) and c >= MIXED_THRESHOLD for p, c in zip(preds, confs)]
        pos       = sum(pos_flags)
        mix       = sum(mix_flags)
        neg       = sum(neg_flags)
        total     = len(preds)
        pos_pct   = (pos / total) * 100 if total > 0 else 0

        if mix > pos and mix > neg:
            box_class = "result-mixed"
            color     = "#facc15"
            verdict   = "MOSTLY MIXED"
            emoji     = "😐"
        elif pos >= neg:
            box_class = "result-positive"
            color     = "#4ade80"
            verdict   = "MOSTLY POSITIVE"
            emoji     = "😊"
        else:
            box_class = "result-negative"
            color     = "#f87171"
            verdict   = "MOSTLY NEGATIVE"
            emoji     = "😞"

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-label" style="color:{color}">Batch Verdict</div>
            <div class="result-emoji">{emoji}</div>
            <div class="result-value">{verdict}</div>
            <div class="result-conf">{pos_pct:.0f}% of reviews are positive</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total",       total)
        m2.metric("😊 Positive", pos)
        m3.metric("😐 Mixed",    mix)
        m4.metric("😞 Negative", neg)

        fig2 = go.Figure(go.Pie(
            labels=['Positive 😊', 'Mixed 😐', 'Negative 😞'],
            values=[pos, mix, neg],
            marker_colors=['#4ade80', '#facc15', '#f87171'],
            hole=0.4
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'},
            height=240, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True, key="pie_chart")

        st.markdown("**🔍 Key Words Per Review:**")
        for rev, pflag, mflag in zip(reviews, pos_flags, mix_flags):
            if mflag:
                sc = "#facc15"
                sl = "😐 Mixed"
            elif pflag:
                sc = "#4ade80"
                sl = "😊 Positive"
            else:
                sc = "#f87171"
                sl = "😞 Negative"
            st.markdown(f"""
            <div style='margin:0.5rem 0; padding:0.7rem 1rem;
                 background:#1e2330; border-radius:10px;
                 border-left:3px solid {sc}'>
                <div style='color:{sc}; font-size:0.78rem;
                     margin-bottom:0.3rem; font-weight:600'>{sl}</div>
                <div style='line-height:2'>{highlight_words(rev)}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder-box">
            <div style='font-size:2.5rem'>📋</div>
            <div style='color:#64748b; margin-top:0.8rem'>
                Enter reviews above and click<br>
                <b style='color:#a78bfa'>Analyze All</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding:1rem 0'>
    Built with ❤️ ·
    <a href='https://github.com/Akhiliny99/Sentiment-Analysis-for-Movie-Reviews' style='color:#a78bfa'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)