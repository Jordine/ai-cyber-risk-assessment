import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(
    page_title="AI Cyber Risk Calculator",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme with better contrast - fixed input text to white
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4a84f5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #4a84f5;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .sub-header {
        font-size: 1.25rem;
        font-weight: 500;
        color: #d1d5db;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #192235;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #d1d5db;
    }
    .highlight {
        background-color: #204080;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 500;
        color: white;
    }
    .result-box {
        background-color: #192235;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4a84f5;
        color: #d1d5db;
    }
    .description {
        font-style: italic;
        color: #a0aec0;
        margin-bottom: 1rem;
    }
    .footnote {
        font-size: 0.8rem;
        color: #a0aec0;
    }
    /* Make sure all text in results is visible */
    .stPlotlyChart text {
        fill: #d1d5db !important;
    }
    /* Fix input text colors - ensure white text */
    .stTextInput input, .stNumberInput input {
        color: white !important;
        background-color: #1E293B !important;
    }
    .stSelectbox > div > div {
        color: white !important;
        background-color: #1E293B !important;
    }
    .stMultiSelect > div > div {
        color: white !important;
        background-color: #1E293B !important;
    }
    /* Ensure slider text is white */
    .stSlider label, .stSlider p {
        color: white !important;
    }
    /* Make sure radio options are white */
    .stRadio label, .stRadio div {
        color: white !important;
    }
    /* Make checkbox labels white */
    .stCheckbox label span {
        color: white !important;
    }
    /* Ensure text in charts is white */
    .js-plotly-plot .plotly .gtitle, .js-plotly-plot .plotly .xtitle, 
    .js-plotly-plot .plotly .ytitle, .js-plotly-plot .plotly .modebar {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">AI Cyber Risk Assessment Tool</div>', unsafe_allow_html=True)
st.markdown("""
This tool uses a Bayesian network model to assess the risk of AI-powered cyber attacks.
Input parameters or use presets based on research to estimate potential economic damage.
""")

# Sidebar for global settings
st.sidebar.markdown("## Global Settings")
show_advanced = st.sidebar.checkbox("Show Advanced Options", value=False)
show_calculations = st.sidebar.checkbox("Show Calculation Details", value=True)
show_formulas = st.sidebar.checkbox("Customize Formulas", value=False)
show_distribution = st.sidebar.checkbox("Show Damage Distribution", value=True)

# Presets for scenarios
st.sidebar.markdown("## Scenario Presets")
preset_options = {
    "Custom (Current Settings)": "custom",
    "Small Budget, Small Targets": "small_small",
    "Small Budget, Large Targets": "small_large",
    "Medium Budget, Mixed Targets": "medium_mixed",
    "Large Budget, Critical Targets": "large_critical",
    "Massive Campaign": "massive"
}

selected_preset = st.sidebar.selectbox(
    "Select a scenario preset:",
    list(preset_options.keys())
)

# Initialize session state for calculated values
if 'n_attacks' not in st.session_state:
    st.session_state.n_attacks = 0
if 'n_spearphish_success' not in st.session_state:
    st.session_state.n_spearphish_success = 0
if 'n_malware_success' not in st.session_state:
    st.session_state.n_malware_success = 0
if 'n_breaches' not in st.session_state:
    st.session_state.n_breaches = 0
if 'total_damage' not in st.session_state:
    st.session_state.total_damage = 0

# Helper function to format currency
def format_currency(value):
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"

# Apply preset if selected
preset_values = {}
if selected_preset != "Custom (Current Settings)":
    preset_code = preset_options[selected_preset]
    
    if preset_code == "small_small":
        preset_values = {
            "budget": 3000,
            "model": "Claude 3.5 Sonnet",
            "company_size": "Small company",
            "security_level": "Standard security",
            "damage_impact": "Small impact",
            "technical_weight": 0.7,
            "susceptibility_level": 0.35,
            "p_malware": 0.175,  # Claude 3.5 Sonnet capability
            "use_distribution": True,
            "distribution_width": 0.2
        }
    elif preset_code == "small_large":
        preset_values = {
            "budget": 3000,
            "model": "Claude 3.5 Sonnet",
            "company_size": "Large company",
            "security_level": "Enterprise security",
            "damage_impact": "Average impact",
            "technical_weight": 0.7,
            "susceptibility_level": 0.35,
            "p_malware": 0.175,  # Claude 3.5 Sonnet capability
            "use_distribution": True,
            "distribution_width": 0.2
        }
    elif preset_code == "medium_mixed":
        preset_values = {
            "budget": 10000,
            "model": "Claude 3.5 Sonnet",
            "company_size": "Medium company",
            "security_level": "Advanced security",
            "damage_impact": "Average impact",
            "technical_weight": 0.7,
            "susceptibility_level": 0.4,
            "p_malware": 0.175,  # Claude 3.5 Sonnet capability
            "use_distribution": True,
            "distribution_width": 0.3
        }
    elif preset_code == "large_critical":
        preset_values = {
            "budget": 50000,
            "model": "Claude 3.5 Sonnet",
            "company_size": "Enterprise",
            "security_level": "Enterprise security",
            "damage_impact": "Severe impact",
            "technical_weight": 0.7,
            "susceptibility_level": 0.5,
            "p_malware": 0.175,  # Claude 3.5 Sonnet capability
            "use_distribution": True,
            "distribution_width": 0.4
        }
    elif preset_code == "massive":
        preset_values = {
            "budget": 100000,
            "model": "Claude 3.5 Sonnet",
            "company_size": "Medium company",
            "security_level": "Standard security",
            "damage_impact": "Average impact",
            "technical_weight": 0.7,
            "susceptibility_level": 0.4,
            "p_malware": 0.175,  # Claude 3.5 Sonnet capability
            "use_distribution": True,
            "distribution_width": 0.5
        }
    
    st.sidebar.markdown(f"**{selected_preset}** parameters applied")

# 1. Number of Attack Attempts
st.markdown('<div class="section-header">1. Number of Attack Attempts</div>', unsafe_allow_html=True)
st.markdown('<div class="description">The total number of spearphishing emails an attacker can send with their budget</div>', unsafe_allow_html=True)

attack_input_method = st.radio(
    "Select input method:",
    ["Direct Input", "Calculate from Budget"]
)

if attack_input_method == "Direct Input":
    n_attacks = st.number_input("Number of attack attempts:", 
                          min_value=1, 
                          max_value=10000000, 
                          value=1000, 
                          step=1000)
    st.session_state.n_attacks = n_attacks
    
else:  # Calculate from Budget
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input("Budget for attacks (USD):", 
                                min_value=10, 
                                max_value=1000000, 
                                value=preset_values.get("budget", 2500), 
                                step=100)
        
        model_options = {
            "Human Expert": 100,
            "Claude 3.5 Haiku": 0.02,
            "Claude 3.5 Sonnet": 0.075,
            "GPT-4o": 0.05,
            "GPT-4o Mini": 0.003,
            "Llama 3.1 405B": 0.0175
        }
        
        selected_model = st.selectbox("Select model for attack generation:", 
                                     list(model_options.keys()),
                                     index=list(model_options.keys()).index(preset_values.get("model", "Claude 3.5 Sonnet")))
        
        default_cost = model_options[selected_model]
        # Allow custom cost input
        use_custom_cost = st.checkbox("Use custom cost per attack")
        
        if use_custom_cost:
            cost_per_attack = st.number_input("Custom cost per attack:", 
                                             min_value=0.0001, 
                                             max_value=1000.0, 
                                             value=float(default_cost),
                                             format="%.4f",
                                             step=0.0001)
        else:
            cost_per_attack = default_cost
        
        # Allow custom token count input
        token_count = st.number_input("Tokens per attack (research + email):", 
                                     min_value=100, 
                                     max_value=100000, 
                                     value=5000, 
                                     step=100)
        
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Cost per attack:** ${cost_per_attack:.4f}")
        st.markdown(f"Based on generating {token_count:,} tokens for research and email composition")
        
        n_attacks = int(budget / cost_per_attack)
        st.markdown(f"**Calculated number of attacks:** {n_attacks:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state.n_attacks = n_attacks

# 2. Probability of Successful Spearphishing
st.markdown('<div class="section-header">2. Probability of Successful Spearphishing</div>', unsafe_allow_html=True)
st.markdown('<div class="description">The probability that a victim clicks the phishing link or opens the malicious attachment (Heiding et al., 2024)</div>', unsafe_allow_html=True)

spearphish_input_method = st.radio(
    "Select spearphishing probability input method:",
    ["Direct Input", "Use Empirical Presets", "Calculate from Benchmarks"]
)

if spearphish_input_method == "Direct Input":
    p_spearphish = st.number_input("Probability of successful spearphishing:", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.12, 
                            step=0.01,
                            format="%.2f")
    
elif spearphish_input_method == "Use Empirical Presets":
    empirical_presets = {
        "Control/Generic Phishing": 0.12,
        "Human Expert Created": 0.54,
        "AI-Automated": 0.54,
        "AI with Human-in-the-Loop": 0.56,
        "Claude 3.5 Sonnet": 0.54,
        "GPT-4o/GPT-4 Turbo": 0.54
    }
    
    selected_preset = st.selectbox("Select empirical preset:", 
                                  list(empirical_presets.keys()))
    
    p_spearphish = empirical_presets[selected_preset]
    
    # Allow custom value override
    use_custom_value = st.checkbox("Use custom success rate")
    if use_custom_value:
        p_spearphish = st.number_input("Custom success rate:", 
                                      min_value=0.0, 
                                      max_value=1.0, 
                                      value=float(p_spearphish),
                                      format="%.2f",
                                      step=0.01)
    
    st.markdown(f"Selected probability: **{p_spearphish:.2f}** ({int(p_spearphish*100)}% success rate)")
    
else:  # Calculate from Benchmarks
    col1, col2 = st.columns(2)
    
    with col1:
        technical_ability_options = {
            "Claude 3.5 Sonnet": 0.90,
            "GPT-4o": 0.90,
            "Claude 3 Opus": 0.80,
            "Claude 3 Haiku": 1.00,
            "GPT-4 Turbo": 1.00,
            "GPT-4o Mini": 1.00,
            "Llama 3.1 (405B)": 1.00,
            "Llama 3.1 (70B)": 1.00,
            "Llama 3.1 (8B)": 0.30,
            "Mixtral (8x7B)": 1.00,
            "Qwen 2 (72B)": 0.20,
            "Human Expert": 0.95
        }
        
        selected_technical = st.selectbox("Select model for technical ability:", 
                                        list(technical_ability_options.keys()),
                                        index=list(technical_ability_options.keys()).index("Claude 3.5 Sonnet"))
        
        default_technical = technical_ability_options[selected_technical]
        
        # Allow manual override of technical ability
        use_custom_technical = st.checkbox("Set custom technical ability")
        if use_custom_technical:
            technical_ability = st.number_input("Custom technical ability score:", 
                                               min_value=0.0, 
                                               max_value=1.0, 
                                               value=float(default_technical),
                                               format="%.2f",
                                               step=0.01)
        else:
            technical_ability = default_technical
        
        susceptibility_level = st.slider("Human susceptibility level:", 
                                        min_value=0.1, 
                                        max_value=0.9, 
                                        value=preset_values.get("susceptibility_level", 0.35), 
                                        step=0.05,
                                        format="%.2f")
        
        # Default weights
        technical_weight = preset_values.get("technical_weight", 0.7)
        susceptibility_weight = 1 - technical_weight
        
        # Allow formula customization
        if show_formulas:
            st.markdown("### Customize Formula Weights")
            technical_weight = st.slider("Technical ability weight:", 
                                       min_value=0.1, 
                                       max_value=0.9, 
                                       value=technical_weight, 
                                       step=0.05,
                                       format="%.2f")
            susceptibility_weight = 1 - technical_weight
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Technical ability score:** {technical_ability:.2f}")
        st.markdown(f"**Susceptibility score:** {susceptibility_level:.2f}")
        
        if show_formulas:
            st.markdown(f"**Technical ability weight:** {technical_weight:.2f}")
            st.markdown(f"**Susceptibility weight:** {susceptibility_weight:.2f}")
        
        # Calculate weighted probability
        p_spearphish = technical_weight * technical_ability + susceptibility_weight * susceptibility_level
        st.markdown(f"**Calculated probability:** {p_spearphish:.2f} ({int(p_spearphish*100)}% success rate)")
        st.markdown(f"Formula: {technical_weight:.2f} × technical_ability + {susceptibility_weight:.2f} × susceptibility")
        st.markdown('</div>', unsafe_allow_html=True)

# Calculate number of successful spearphishing attempts
n_spearphish_success = int(st.session_state.n_attacks * p_spearphish)
st.session_state.n_spearphish_success = n_spearphish_success

if show_calculations:
    st.markdown(f"**Successful spearphishing attempts:** {n_spearphish_success:,} of {st.session_state.n_attacks:,} attacks")

# 3. Probability of Successful Malware Development
st.markdown('<div class="section-header">3. Probability of Successful Malware Development</div>', unsafe_allow_html=True)
st.markdown('<div class="description">The probability that functional malware is successfully created by the LLM (based on CyberBench metrics)</div>', unsafe_allow_html=True)

malware_input_method = st.radio(
    "Select malware development probability input method:",
    ["Direct Input", "Use Capability Presets", "Calculate from Benchmarks"]
)

if malware_input_method == "Direct Input":
    p_malware = st.number_input("Probability of successful malware development:", 
                         min_value=0.0, 
                         max_value=1.0, 
                         value=preset_values.get("p_malware", 0.1), 
                         step=0.01,
                         format="%.2f")
    
elif malware_input_method == "Use Capability Presets":
    capability_presets = {
        "Claude 3.5 Sonnet": 0.175,
        "GPT-4o": 0.125,
        "Claude 3 Opus": 0.100,
        "OpenAI o1-preview": 0.100,
        "Llama 3.1 405B": 0.075,
        "Mixtral 8x22B": 0.075,
        "Gemini 1.5 Pro": 0.075,
        "Llama 3 70B": 0.050
    }
    
    selected_capability = st.selectbox("Select capability preset:", 
                                      list(capability_presets.keys()),
                                      index=list(capability_presets.keys()).index("Claude 3.5 Sonnet"))
    
    default_capability = capability_presets[selected_capability]
    
    # Allow manual override
    use_custom_capability = st.checkbox("Set custom capability")
    if use_custom_capability:
        p_malware = st.number_input("Custom capability score:", 
                                   min_value=0.0, 
                                   max_value=1.0, 
                                   value=float(default_capability),
                                   format="%.3f",
                                   step=0.001)
    else:
        p_malware = default_capability
    
    st.markdown(f"Selected probability: **{p_malware:.3f}** ({p_malware*100:.1f}% capability)")
    st.markdown('<div class="footnote">Based on CyberBench (Anurin et al., 2024)</div>', unsafe_allow_html=True)
    
else:  # Calculate from Benchmarks
    col1, col2 = st.columns(2)
    
    with col1:
        generation_score = st.slider("Malware generation score:", 
                                   min_value=0.0, 
                                   max_value=1.0, 
                                   value=0.15, 
                                   step=0.01,
                                   format="%.2f")
        
        code_quality = st.slider("Code quality score:", 
                               min_value=0.0, 
                               max_value=1.0, 
                               value=0.2, 
                               step=0.01,
                               format="%.2f")
        
        # Default weights
        generation_weight = 0.6
        quality_weight = 0.4
        
        # Allow formula customization
        if show_formulas:
            st.markdown("### Customize Formula Weights")
            generation_weight = st.slider("Generation score weight:", 
                                        min_value=0.1, 
                                        max_value=0.9, 
                                        value=0.6, 
                                        step=0.05,
                                        format="%.2f")
            quality_weight = 1 - generation_weight
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Generation score:** {generation_score:.2f}")
        st.markdown(f"**Code quality score:** {code_quality:.2f}")
        
        if show_formulas:
            st.markdown(f"**Generation weight:** {generation_weight:.2f}")
            st.markdown(f"**Quality weight:** {quality_weight:.2f}")
        
        # Calculate weighted probability
        p_malware = generation_weight * generation_score + quality_weight * code_quality
        st.markdown(f"**Calculated probability:** {p_malware:.3f} ({p_malware*100:.1f}% capability)")
        st.markdown(f"Formula: {generation_weight:.2f} × generation_score + {quality_weight:.2f} × code_quality")
        st.markdown('</div>', unsafe_allow_html=True)

# Calculate number of successful malware developments
n_malware_success = int(st.session_state.n_spearphish_success * p_malware)
st.session_state.n_malware_success = n_malware_success

if show_calculations:
    st.markdown(f"**Successful malware developments:** {n_malware_success:,} of {st.session_state.n_spearphish_success:,} attempts")

# 4. Probability of Successful Persistence
st.markdown('<div class="section-header">4. Probability of Successful Persistence</div>', unsafe_allow_html=True)
st.markdown('<div class="description">The probability that the malware evades detection and achieves persistence in the victim\'s system</div>', unsafe_allow_html=True)

persistence_input_method = st.radio(
    "Select persistence probability input method:",
    ["Direct Input", "Use Security Level Presets"]
)

if persistence_input_method == "Direct Input":
    p_persistence = st.number_input("Probability of successful persistence:", 
                             min_value=0.0, 
                             max_value=1.0, 
                             value=0.25, 
                             step=0.01,
                             format="%.2f")
    
else:  # Use Security Level Presets
    security_presets = {
        "Basic security": 0.40,
        "Standard security": 0.25,
        "Advanced security": 0.15,
        "Enterprise security": 0.08
    }
    
    selected_security = st.selectbox("Select security level preset:", 
                                    list(security_presets.keys()),
                                    index=list(security_presets.keys()).index(preset_values.get("security_level", "Standard security")))
    
    default_persistence = security_presets[selected_security]
    
    # Allow manual override
    use_custom_persistence = st.checkbox("Set custom persistence rate")
    if use_custom_persistence:
        p_persistence = st.number_input("Custom persistence probability:", 
                                      min_value=0.0, 
                                      max_value=1.0, 
                                      value=float(default_persistence),
                                      format="%.2f",
                                      step=0.01)
    else:
        p_persistence = default_persistence
    
    # Allow formula customization if needed
    if show_formulas:
        st.markdown("### Formula")
        st.markdown("P(persistence) = 1 - detection_rate")
        st.markdown(f"Current detection rate: {1-p_persistence:.2f}")
        
        # Add option to manually set detection rate
        use_detection_rate = st.checkbox("Set detection rate directly")
        if use_detection_rate:
            detection_rate = st.slider("Detection rate:", 
                                     min_value=0.0, 
                                     max_value=1.0, 
                                     value=1-p_persistence, 
                                     step=0.01,
                                     format="%.2f")
            p_persistence = 1 - detection_rate
    
    detection_rate = 1 - p_persistence
    st.markdown(f"Selected probability: **{p_persistence:.2f}** (Detection rate: {detection_rate:.2f} or {int(detection_rate*100)}%)")

# Calculate number of successful breaches
n_breaches = int(st.session_state.n_malware_success * p_persistence)
st.session_state.n_breaches = n_breaches

if show_calculations:
    st.markdown(f"**Successful breaches:** {n_breaches:,} of {st.session_state.n_malware_success:,} attempts")

# 5. Severity per Successful Attack
st.markdown('<div class="section-header">5. Severity per Successful Attack</div>', unsafe_allow_html=True)
st.markdown('<div class="description">The estimated financial damage caused by each successful breach</div>', unsafe_allow_html=True)

severity_input_method = st.radio(
    "Select severity input method:",
    ["Direct Input", "Calculate from Company Value"]
)

if severity_input_method == "Direct Input":
    severity = st.number_input("Severity per successful attack (USD):", 
                              min_value=1000, 
                              max_value=100000000, 
                              value=4000000, 
                              step=10000)
    
else:  # Calculate from Company Value
    col1, col2 = st.columns(2)
    
    with col1:
        company_value_options = {
            "Small company": 10000000,  # $10M
            "Medium company": 100000000,  # $100M
            "Large company": 1000000000,  # $1B
            "Enterprise": 10000000000  # $10B
        }
        
        selected_company = st.selectbox("Select company size:", 
                                      list(company_value_options.keys()),
                                      index=list(company_value_options.keys()).index(preset_values.get("company_size", "Medium company")))
        
        default_company_value = company_value_options[selected_company]
        
        # Allow custom company value
        use_custom_company = st.checkbox("Set custom company value")
        if use_custom_company:
            company_value = st.number_input("Custom company value (USD):", 
                                          min_value=100000, 
                                          max_value=100000000000, 
                                          value=int(default_company_value),
                                          step=1000000)
        else:
            company_value = default_company_value
        
        damage_percentage_options = {
            "Small impact": 0.0005,  # 0.05%
            "Average impact": 0.02,  # 2%
            "Severe impact": 0.1,    # 10%
            "Catastrophic impact": 0.4  # 40%
        }
        
        selected_damage = st.selectbox("Select damage impact level:", 
                                     list(damage_percentage_options.keys()),
                                     index=list(damage_percentage_options.keys()).index(preset_values.get("damage_impact", "Average impact")))
        
        default_damage_percentage = damage_percentage_options[selected_damage]
        
        # Allow custom damage percentage
        use_custom_damage = st.checkbox("Set custom damage percentage")
        if use_custom_damage:
            damage_percentage = st.number_input("Custom damage percentage:", 
                                              min_value=0.0001, 
                                              max_value=1.0, 
                                              value=float(default_damage_percentage),
                                              format="%.4f",
                                              step=0.0001)
        else:
            damage_percentage = default_damage_percentage
        
        # Allow formula customization
        if show_formulas:
            st.markdown("### Formula")
            st.markdown("Severity = company_value × damage_percentage")
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Company value:** {format_currency(company_value)}")
        st.markdown(f"**Damage percentage:** {damage_percentage:.2%}")
        
        # Calculate severity
        severity = company_value * damage_percentage
        st.markdown(f"**Calculated severity:** {format_currency(severity)} per breach")
        st.markdown("Formula: company_value × damage_percentage")
        st.markdown('</div>', unsafe_allow_html=True)

# Calculate total damage
total_damage = st.session_state.n_breaches * severity
st.session_state.total_damage = total_damage

# Distribution settings
if show_distribution:
    st.markdown('<div class="section-header">Damage Distribution Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Configure the variability in potential damages to see a range of possible outcomes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        distribution_type = st.selectbox(
            "Distribution type:",
            ["Normal", "Log-normal", "Triangular"]
        )
        
        # Width parameter (standard deviation, etc.)
        distribution_width = st.slider(
            "Distribution width factor:", 
            min_value=0.05, 
            max_value=1.0, 
            value=preset_values.get("distribution_width", 0.2), 
            step=0.05,
            format="%.2f"
        )
        
        n_simulations = st.slider(
            "Number of simulations:", 
            min_value=100, 
            max_value=10000, 
            value=1000, 
            step=100
        )
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Distribution Parameters:**")
        
        if distribution_type == "Normal":
            st.markdown(f"- Mean: {format_currency(total_damage)}")
            st.markdown(f"- Standard Deviation: {format_currency(total_damage * distribution_width)}")
            st.markdown("- Normal distribution simulates symmetric variation around the expected damage value")
        elif distribution_type == "Log-normal":
            st.markdown(f"- Median: {format_currency(total_damage)}")
            st.markdown(f"- Sigma: {distribution_width}")
            st.markdown("- Log-normal distribution simulates higher probability of extreme values (heavy right tail)")
        else:  # Triangular
            min_val = total_damage * (1 - distribution_width)
            max_val = total_damage * (1 + distribution_width * 2)  # Asymmetric, more upside risk
            st.markdown(f"- Minimum: {format_currency(min_val)}")
            st.markdown(f"- Most likely: {format_currency(total_damage)}")
            st.markdown(f"- Maximum: {format_currency(max_val)}")
            st.markdown("- Triangular distribution simulates a defined range of possible values")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display results
st.markdown('<div class="section-header">Risk Assessment Results</div>', unsafe_allow_html=True)

# Remove empty box by using columns layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Attack Chain Summary</div>', unsafe_allow_html=True)
    st.markdown(f"• **Total attack attempts:** {st.session_state.n_attacks:,}")
    st.markdown(f"• **Successful spearphishing:** {st.session_state.n_spearphish_success:,} ({p_spearphish:.1%})")
    st.markdown(f"• **Successful malware development:** {st.session_state.n_malware_success:,} ({p_malware:.1%})")
    st.markdown(f"• **Successful breaches:** {st.session_state.n_breaches:,} ({p_persistence:.1%})")
    st.markdown(f"• **Severity per breach:** {format_currency(severity)}")
    st.markdown(f"• **Total estimated damage:** <span class='highlight'>{format_currency(total_damage)}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Create a funnel chart to visualize the attack chain
    fig = go.Figure(go.Funnel(
        y = ['Attack Attempts', 'Successful Phishing', 'Successful Malware', 'Successful Breaches'],
        x = [st.session_state.n_attacks, st.session_state.n_spearphish_success, 
             st.session_state.n_malware_success, st.session_state.n_breaches],
        textinfo = "value+percent initial",
        textfont = {"color": "white"}
    ))
    
    fig.update_layout(
        title="Attack Funnel Visualization",
        height=400,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Generate and display damage distribution
if show_distribution and n_breaches > 0:
    st.markdown('<div class="section-header">Damage Distribution Analysis</div>', unsafe_allow_html=True)
    
    # Create distribution
    np.random.seed(42)  # For reproducibility
    
    if distribution_type == "Normal":
        damage_samples = np.random.normal(
            loc=total_damage, 
            scale=total_damage * distribution_width, 
            size=n_simulations
        )
        # Ensure no negative values
        damage_samples = np.maximum(damage_samples, 0)
        
    elif distribution_type == "Log-normal":
        # For log-normal, we need to convert parameters
        # We want the median of the distribution to be our total_damage
        mu = np.log(total_damage)
        sigma = distribution_width
        damage_samples = np.random.lognormal(
            mean=mu, 
            sigma=sigma, 
            size=n_simulations
        )
        
    else:  # Triangular
        min_val = max(0, total_damage * (1 - distribution_width))
        mode_val = total_damage
        max_val = total_damage * (1 + distribution_width * 2)  # More upside risk
        
        damage_samples = np.random.triangular(
            left=min_val,
            mode=mode_val,
            right=max_val,
            size=n_simulations
        )
    
    # Calculate statistics
    mean_damage = np.mean(damage_samples)
    median_damage = np.median(damage_samples)
    p05_damage = np.percentile(damage_samples, 5)
    p95_damage = np.percentile(damage_samples, 95)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create histogram
        fig = px.histogram(
            damage_samples, 
            nbins=50, 
            labels={'value': 'Potential Damage ($)', 'count': 'Frequency'},
            title="Distribution of Potential Damages",
            template="plotly_dark"
        )
        
        # Add vertical lines for statistics
        fig.add_vline(x=total_damage, line_width=2, line_dash="dash", line_color="red", 
                     annotation_text="Expected Value", annotation_position="top right", annotation_font_color="red")
        fig.add_vline(x=p05_damage, line_width=1, line_dash="dot", line_color="yellow", 
                     annotation_text="5th Percentile", annotation_position="bottom left", annotation_font_color="yellow")
        fig.add_vline(x=p95_damage, line_width=1, line_dash="dot", line_color="green", 
                     annotation_text="95th Percentile", annotation_position="top right", annotation_font_color="green")
        
        fig.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Distribution Statistics")
        
        st.markdown(f"**Expected value:** {format_currency(total_damage)}")
        st.markdown(f"**Mean value:** {format_currency(mean_damage)}")
        st.markdown(f"**Median value:** {format_currency(median_damage)}")
        st.markdown(f"**5th percentile:** {format_currency(p05_damage)}")
        st.markdown(f"**95th percentile:** {format_currency(p95_damage)}")
        
        # Calculate Value at Risk
        VaR_95 = np.percentile(damage_samples, 95)
        st.markdown(f"**Value at Risk (95%):** {format_currency(VaR_95)}")
        
        st.markdown("### Interpretation")
        st.markdown(f"There is a 90% probability that damages will fall between {format_currency(p05_damage)} and {format_currency(p95_damage)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Add sensitivity analysis for advanced users
if show_advanced:
    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">How the estimated damage changes with different parameter values</div>', unsafe_allow_html=True)
    
    # Create parameter ranges
    p_spearphish_range = np.linspace(max(0.01, p_spearphish-0.2), min(0.99, p_spearphish+0.2), 5)
    p_malware_range = np.linspace(max(0.01, p_malware-0.1), min(0.99, p_malware+0.1), 5)
    p_persistence_range = np.linspace(max(0.01, p_persistence-0.1), min(0.99, p_persistence+0.1), 5)
    
    param_to_analyze = st.selectbox(
        "Select parameter for sensitivity analysis:", 
        ["Spearphishing probability", "Malware development probability", "Persistence probability"]
    )
    
    if param_to_analyze == "Spearphishing probability":
        sensitivity_data = []
        for p in p_spearphish_range:
            n_spear = int(st.session_state.n_attacks * p)
            n_malw = int(n_spear * p_malware)
            n_breach = int(n_malw * p_persistence)
            damage = n_breach * severity
            sensitivity_data.append({
                'Parameter Value': f"{p:.2f}", 
                'Total Damage': damage,
                'Breaches': n_breach
            })
        
        param_name = "Spearphishing probability"
        
    elif param_to_analyze == "Malware development probability":
        sensitivity_data = []
        for p in p_malware_range:
            n_spear = st.session_state.n_spearphish_success
            n_malw = int(n_spear * p)
            n_breach = int(n_malw * p_persistence)
            damage = n_breach * severity
            sensitivity_data.append({
                'Parameter Value': f"{p:.2f}", 
                'Total Damage': damage,
                'Breaches': n_breach
            })
        
        param_name = "Malware probability"
        
    else:  # Persistence probability
        sensitivity_data = []
        for p in p_persistence_range:
            n_spear = st.session_state.n_spearphish_success
            n_malw = st.session_state.n_malware_success
            n_breach = int(n_malw * p)
            damage = n_breach * severity
            sensitivity_data.append({
                'Parameter Value': f"{p:.2f}", 
                'Total Damage': damage,
                'Breaches': n_breach
            })
        
        param_name = "Persistence probability"
    
    # Create sensitivity visualization
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    fig = px.bar(sensitivity_df, 
                x='Parameter Value', 
                y='Total Damage',
                text='Breaches',
                title=f"Sensitivity Analysis - Impact of {param_name}",
                labels={'Total Damage': 'Estimated Total Damage ($)'},
                height=400,
                template="plotly_dark")
    
    fig.update_traces(texttemplate='%{text:,} breaches', textposition='outside', textfont_color="white")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Show the full Bayesian network formula if customization is enabled
if show_formulas:
    st.markdown('<div class="section-header">Complete Model Formula</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**Bayesian Network Formula:**")
    st.markdown("""
    1. N_successful_spearphishing = N_attacks × P(spearphishing)
    2. N_successful_malware = N_successful_spearphishing × P(malware)
    3. N_successful_breaches = N_successful_malware × P(persistence)
    4. Total_damage = N_successful_breaches × Severity_per_breach
    """)
    
    st.markdown("**Current Weights:**")
    
    if spearphish_input_method == "Calculate from Benchmarks":
        st.markdown(f"P(spearphishing) = {technical_weight:.2f} × technical_ability + {susceptibility_weight:.2f} × susceptibility")
    
    if malware_input_method == "Calculate from Benchmarks":
        st.markdown(f"P(malware) = {generation_weight:.2f} × generation_score + {quality_weight:.2f} × code_quality")
    
    if persistence_input_method == "Use Security Level Presets":
        st.markdown(f"P(persistence) = 1 - {1-p_persistence:.2f} (detection_rate)")
    
    if severity_input_method == "Calculate from Company Value":
        st.markdown(f"Severity = {format_currency(company_value)} × {damage_percentage:.2%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add footer with references
st.markdown("---")
st.markdown("""
**References:**
- Zhang et al. (2024). Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models.
- Heiding et al. (2024). Evaluating Large Language Models' Capability to Launch Fully Automated Spear Phishing Campaigns.
- Anurin et al. (2024). Catastrophic Cyber Capabilities Benchmark (3CB): Robustly Evaluating LLM Agent Cyber Offense Capabilities.
- IBM. (2023). Cost of a Data Breach Report.
- Ponemon Institute. (2023). Cost of Cybercrime Study.
""")