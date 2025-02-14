import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Charger le modèle KNN et le scaler
model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")


# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Maladies Cardiaques ❤️",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styles CSS améliorés
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Style global */
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f0f2f5;
            color: #1a1a1a;
        }
        
        /* Nouveau Header avec effet glassmorphism */
        .header-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin: -1rem -1rem 2rem -1rem;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        }
        
        .header-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }
        
        .logo-pulse {
            animation: pulse 2s infinite;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #ff4b6b, #7450fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            font-weight: 400;
            max-width: 600px;
            line-height: 1.6;
        }
        
        /* Cards modernes */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        
        /* Sections */
        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 1rem;
            padding-left: 0.5rem;
            border-left: 4px solid #ff4b6b;
        }
        
        /* Inputs stylisés */
        .stSlider > div > div > div > div {
            background-color: #ff4b6b !important;
        }
        
        .stSelectbox > div > div {
            background: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        
        /* Bouton moderne */
        .stButton > button {
            background: linear-gradient(45deg, #ff4b6b, #7450fe);
            color: white;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            border: none;
            font-weight: 500;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(255, 75, 107, 0.3);
        }
        
        /* Résultats */
        .result-card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .result-positive {
            background: linear-gradient(45deg, #ff4b6b, #ff8080);
            color: white;
        }
        
        .result-negative {
            background: linear-gradient(45deg, #2ecc71, #26de81);
            color: white;
        }
        
        /* Footer moderne */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
            margin-top: 3rem;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            backdrop-filter: blur(5px);
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            text-align: center;
            padding: 5px 10px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Animation de chargement */
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        .loading {
            animation: shimmer 2s infinite linear;
            background: linear-gradient(to right, #f6f7f8 0%, #edeef1 20%, #f6f7f8 40%, #f6f7f8 100%);
            background-size: 1000px 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Nouveau Header
st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <div class="logo-pulse">💖</div>
            <h1 class="title">Prédiction des Maladies Cardiaques</h1>
            <p class="subtitle">
                Un outil intelligent d'aide à la décision médicale utilisant l'intelligence artificielle 
                pour évaluer les risques cardiaques avec précision
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Information Card
st.markdown("""
    <div class="card">
        <div class="section-title">ℹ️ Guide d'utilisation</div>
        <ol style="color: #666; line-height: 1.6;">
            <li>Remplissez vos informations médicales dans les champs ci-dessous</li>
            <li>Vérifiez que toutes les valeurs sont correctement saisies</li>
            <li>Cliquez sur le bouton "Analyser les risques" pour obtenir une évaluation</li>
            <li>Consultez les résultats et les recommandations personnalisées</li>
        </ol>
        <div style="background: #fff3f4; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong style="color: #ff4b6b;">⚠️ Note importante:</strong>
            <span style="color: #666;"> Cet outil est uniquement à titre indicatif et ne remplace pas l'avis d'un professionnel de santé.</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Formulaire en 3 colonnes
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Informations personnelles</div>', unsafe_allow_html=True)
    age = st.slider("Âge", 20, 100, 50)
    sex = st.selectbox("Sexe", [("Homme 👨", 1), ("Femme 👩", 0)])[1]
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Mesures principales</div>', unsafe_allow_html=True)
    trestbps = st.slider("Pression artérielle (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholestérol (mg/dl)", 100, 400, 200)
    thalach = st.slider("Fréquence cardiaque max", 60, 220, 150)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Indicateurs spécifiques</div>', unsafe_allow_html=True)
    cp = st.selectbox("Type de douleur thoracique", [
        ("Angine typique", 0),
        ("Angine atypique", 1),
        ("Douleur non angineuse", 2),
        ("Asymptomatique", 3)
    ])[1]
    fbs = st.selectbox("Glycémie à jeun > 120 mg/dl ?", [("Non 🟢", 0), ("Oui 🔴", 1)])[1]
    st.markdown('</div>', unsafe_allow_html=True)

# Paramètres avancés
with st.expander("🔬 Paramètres avancés"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    
    with col4:
        restecg = st.selectbox("Résultats électrocardiographiques", [
            ("Normal", 0),
            ("Anomalie ST-T", 1),
            ("Hypertrophie ventriculaire", 2)
        ])[1]
        exang = st.selectbox("Angine induite par l'exercice ?", [("Non 🟢", 0), ("Oui 🔴", 1)])[1]
        oldpeak = st.slider("Dépression ST", 0.0, 6.0, 1.0, 0.1)

    with col5:
        slope = st.selectbox("Pente du segment ST", [
            ("Ascendante", 0),
            ("Plate", 1),
            ("Descendante", 2)
        ])[1]
        ca = st.slider("Nombre de vaisseaux majeurs colorés", 0, 3, 1)
        thal = st.selectbox("Thalassémie", [
            ("Normal", 3),
            ("Défaut fixé", 6),
            ("Défaut réversible", 7)
        ])[1]
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton d'analyse
st.markdown('<div class="card">', unsafe_allow_html=True)
if st.button("🔍 Analyser les risques", help="Cliquez pour obtenir une prédiction"):
    # Animation de chargement
    with st.spinner("Analyse en cours..."):
        # Préparation des données
        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                              exang, oldpeak, slope, ca, thal]])
        user_input_scaled = scaler.transform(user_input)
        
        # Prédiction
        prediction = model.predict(user_input_scaled)
        proba = model.predict_proba(user_input_scaled)[0]

        # Affichage du résultat
        if prediction[0] == 1:
            st.markdown(
                f'<div class="result-card result-positive">'
                f'<h2 style="margin-bottom: 1rem;">🚨 Risque Cardiovasculaire Détecté</h2>'
                f'<p style="font-size: 1.2rem;">Probabilité de risque: <strong>{proba[1]:.1%}</strong></p>'
                f'<div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 8px;">'
                f'<strong>Recommandation:</strong> Une consultation médicale est fortement conseillée'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-card result-negative">'
                f'<h2 style="margin-bottom: 1rem;">✅ Risque Cardiovasculaire Faible</h2>'
                f'<p style="font-size: 1.2rem;">Probabilité de risque: <strong>{proba[1]:.1%}</strong></p>'
                f'<div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 8px;">'
                f'<strong>Recommandation:</strong> Continuez à maintenir de bonnes habitudes de santé'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Graphique de la probabilité avec style amélioré
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = proba[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Indicateur de risque", 'font': {'size': 24, 'family': 'Poppins'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(255,75, 107, 0.8)"},
                'steps': [
                    {'range': [0, 33], 'color': "rgba(46, 204, 113, 0.3)"},
                    {'range': [33, 66], 'color': "rgba(241, 196, 15, 0.3)"},
                    {'range': [66, 100], 'color': "rgba(255, 75, 107, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': proba[1] * 100
                }
            }
        ))
        
        # Style du graphique
        fig.update_layout(
            height=300,
            font={'family': "Poppins"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=40, b=0, l=0, r=0)
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer moderne
st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">💖</span>
        </div>
        <div style="margin-bottom: 0.5rem; font-weight: 500; color: #333;">
            Développé avec passion par Fatao OUEDRAOGO & Ibrahim OUEDRAOGO
        </div>
        <div style="color: #666; font-size: 0.9rem;">
            Version 2.0 | © 2024 Tous droits réservés
        </div>
    </div>
""", unsafe_allow_html=True)