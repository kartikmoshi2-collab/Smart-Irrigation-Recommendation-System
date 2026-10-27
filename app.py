import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Irrigation | Field Console",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Visual style: restrained, dashboard-like, and easy to explain in a viva ---
st.markdown("""
<style>
:root { --green: #246b4b; --green-dark: #174b36; --mint: #e9f4ed; --ink: #1d2b25; }
.stApp { background: #f6f8f5; color: var(--ink); }
[data-testid="stHeader"] { background: rgba(246,248,245,.92); }
.block-container { padding-top: 1.6rem; padding-bottom: 2.4rem; max-width: 1280px; }
h1, h2, h3 { color: #173f2d; letter-spacing: -0.02em; }
.hero {
  padding: 1.35rem 1.5rem; border-radius: 18px;
  background: linear-gradient(120deg, #173f2d 0%, #246b4b 72%, #4e8b64 100%);
  color: white; margin-bottom: 1.2rem;
}
.hero h1 { color: white; margin: 0; font-size: 2rem; }
.hero p { color: #e2eee6; margin: .45rem 0 0 0; font-size: 1rem; }
.kpi {
  background: white; border: 1px solid #e1e9e2; border-radius: 15px;
  padding: 1rem 1.1rem; min-height: 120px;
}
.kpi-label { color: #66756c; font-size: .82rem; text-transform: uppercase; letter-spacing: .07em; }
.kpi-value { color: #173f2d; font-size: 1.8rem; font-weight: 750; margin-top: .25rem; }
.kpi-note { color: #66756c; font-size: .83rem; margin-top: .15rem; }
.panel {
  background: white; border: 1px solid #e1e9e2; border-radius: 15px;
  padding: 1.1rem 1.2rem;
}
.small-muted { color: #68776e; font-size: .88rem; }
div.stButton > button[kind="primary"] {
  background: #246b4b; border: 1px solid #246b4b; border-radius: 10px;
  min-height: 2.8rem; font-weight: 650;
}
div.stButton > button[kind="primary"]:hover { background: #174b36; border-color: #174b36; }
[data-testid="stSidebar"] { background: #edf3ee; }
hr { border-color: #e1e9e2; }
</style>
""", unsafe_allow_html=True)

# --- Mamdani fuzzy controller from the project's report ---
def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if b == a else (x - a) / (b - a)
    right = np.ones_like(x) if c == b else (c - x) / (c - b)
    return np.clip(np.minimum(left, right), 0.0, 1.0)

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if b == a else (x - a) / (b - a)
    right = np.ones_like(x) if d == c else (d - x) / (d - c)
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0.0, 1.0)

MOIST = {
    "Dry": ("trap", 0, 0, 15, 38),
    "Moist": ("tri", 20, 42, 62),
    "Wet": ("trap", 48, 62, 100, 100),
}
TEMP = {
    "Cool": ("trap", 0, 0, 15, 25),
    "Warm": ("tri", 15, 25, 35),
    "Hot": ("trap", 25, 35, 50, 50),
}
HUMID = {
    "Low": ("trap", 0, 0, 25, 45),
    "Medium": ("tri", 25, 50, 75),
    "High": ("trap", 55, 75, 100, 100),
}
OUT = {
    "None": ("trap", 0, 0, 2, 6),
    "Low": ("tri", 4, 12, 22),
    "Medium": ("tri", 15, 25, 38),
    "High": ("trap", 30, 42, 60, 60),
}
T_SCORE = {"Cool": 0, "Warm": 1, "Hot": 2}
H_SCORE = {"High": 0, "Medium": 1, "Low": 2}
Y = np.linspace(0, 60, 601)
OUT_MF = {
    name: (trimf(Y, *spec[1:]) if spec[0] == "tri" else trapmf(Y, *spec[1:]))
    for name, spec in OUT.items()
}

def membership(spec, x):
    return trimf(x, *spec[1:]) if spec[0] == "tri" else trapmf(x, *spec[1:])

def consequent(m, t, h):
    demand = T_SCORE[t] + H_SCORE[h]
    if m == "Dry":
        return "Medium" if demand <= 1 else "High"
    if m == "Moist":
        return "None" if demand <= 3 else "Low"
    return "None"

RULES = [(m, t, h, consequent(m, t, h)) for m in MOIST for t in TEMP for h in HUMID]

def fuzzy_irrigation(M, T, H):
    M = float(np.clip(M, 0, 100))
    T = float(np.clip(T, 0, 50))
    H = float(np.clip(H, 0, 100))
    mu_m = {k: float(membership(v, M)) for k, v in MOIST.items()}
    mu_t = {k: float(membership(v, T)) for k, v in TEMP.items()}
    mu_h = {k: float(membership(v, H)) for k, v in HUMID.items()}
    agg = np.zeros_like(Y)
    fired = []
    for m, t, h, e in RULES:
        strength = min(mu_m[m], mu_t[t], mu_h[h])
        if strength > 0:
            agg = np.maximum(agg, np.minimum(strength, OUT_MF[e]))
            fired.append((m, t, h, e, strength))
    if agg.sum() < 1e-9:
        return 0.0, mu_m, mu_t, mu_h, fired, agg
    duration = float((Y * agg).sum() / agg.sum())
    if duration < 4.0:
        duration = 0.0
    return duration, mu_m, mu_t, mu_h, fired, agg

def condition_label(moisture):
    if moisture < 30:
        return "Very dry", "Soil moisture is low. Check the reading and consider irrigation."
    if moisture < 40:
        return "Dry", "The soil is on the dry side; the fuzzy controller will consider weather conditions."
    if moisture <= 65:
        return "In target band", "Moisture is within the project's target band of 40–65%."
    if moisture <= 75:
        return "Moist", "Moisture is above the target band; irrigation may not be needed."
    return "Wet", "The soil is very moist. The controller generally recommends no irrigation."

# --- Sidebar inputs ---
with st.sidebar:
    st.markdown("## 🌿 Field readings")
    st.caption("Enter current sensor values to calculate a recommendation.")
    moisture = st.slider("Soil moisture (%)", 0, 100, 28, help="Project input range: 0–100%.")
    temperature = st.slider("Air temperature (°C)", 0, 50, 32, help="Project input range: 0–50°C.")
    humidity = st.slider("Relative humidity (%)", 0, 100, 38, help="Project input range: 0–100%.")
    st.markdown("---")
    calculate = st.button("Calculate recommendation", type="primary", use_container_width=True)
    st.caption("The controller uses the report's 27-rule Mamdani fuzzy system. It is a software demo, not a connected irrigation valve.")

# Persist latest values naturally as Streamlit reruns on interaction
duration, mu_m, mu_t, mu_h, fired, aggregate = fuzzy_irrigation(moisture, temperature, humidity)
condition, condition_note = condition_label(moisture)

st.markdown("""
<div class="hero">
  <div style="font-size:.78rem; text-transform:uppercase; letter-spacing:.13em; opacity:.82;">Soft Computing Project · Mamdani Fuzzy Logic</div>
  <h1>Smart Irrigation Recommendation System</h1>
  <p>A practical dashboard for turning soil and weather readings into an explainable watering recommendation.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Field overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Soil moisture</div><div class="kpi-value">{moisture}%</div><div class="kpi-note">Current input</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Temperature</div><div class="kpi-value">{temperature}°C</div><div class="kpi-note">Current input</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Relative humidity</div><div class="kpi-value">{humidity}%</div><div class="kpi-note">Current input</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Recommended duration</div><div class="kpi-value">{duration:.1f} min</div><div class="kpi-note">Per decision cycle</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Irrigation recommendation")
    if duration == 0:
        st.success("No irrigation recommended right now")
        st.markdown("**Suggested action:** Keep monitoring the readings and reassess at the next decision point.")
    else:
        st.info(f"Recommended watering time: **{duration:.1f} minutes**")
        st.markdown("**Suggested action:** Use this duration as a software recommendation for the next irrigation cycle.")
    st.progress(min(max(duration / 60.0, 0.0), 1.0), text=f"{duration:.1f} of 60 minutes maximum")
    st.markdown(f"**Soil status:** {condition}")
    st.caption(condition_note)
    st.markdown('<div class="small-muted">The project applies a 4-minute minimum run-time: computed values below 4 minutes are changed to zero.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### How the system reached this result")
    st.write("The controller converts each input into fuzzy membership degrees, evaluates all 27 rules, combines their outputs, then uses centroid defuzzification.")
    membership_rows = []
    for name, val in mu_m.items():
        membership_rows.append({"Input": "Moisture", "Fuzzy set": name, "Membership": val})
    for name, val in mu_t.items():
        membership_rows.append({"Input": "Temperature", "Fuzzy set": name, "Membership": val})
    for name, val in mu_h.items():
        membership_rows.append({"Input": "Humidity", "Fuzzy set": name, "Membership": val})
    mf_df = pd.DataFrame(membership_rows)
    st.dataframe(
        mf_df.style.format({"Membership": "{:.2f}"}).bar(subset=["Membership"], vmin=0, vmax=1, color="#a8d5b8"),
        hide_index=True, use_container_width=True
    )
    st.caption(f"{len(fired)} of the 27 rules have a non-zero firing strength for these readings.")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3 = st.tabs(["Output curve", "Membership functions", "Rule trace"])

with tab1:
    st.markdown("### Aggregated output set")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(Y, aggregate, linewidth=2.5, color="#246b4b", label="Aggregated fuzzy output")
    ax.fill_between(Y, aggregate, color="#8bc4a0", alpha=0.35)
    ax.axvline(duration, color="#c17836", linestyle="--", linewidth=1.8, label=f"Recommendation: {duration:.1f} min")
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Irrigation duration (minutes)"); ax.set_ylabel("Membership degree")
    ax.grid(alpha=.2); ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("The orange marker shows the crisp duration after centroid defuzzification and the 4-minute minimum-run rule.")

with tab2:
    st.markdown("### Fuzzy membership functions")
    variable_choice = st.selectbox("Choose a variable", ["Soil moisture (%)", "Temperature (°C)", "Relative humidity (%)", "Irrigation duration (minutes)"])
    if variable_choice == "Soil moisture (%)":
        specs, x = MOIST, np.linspace(0, 100, 401)
    elif variable_choice == "Temperature (°C)":
        specs, x = TEMP, np.linspace(0, 50, 401)
    elif variable_choice == "Relative humidity (%)":
        specs, x = HUMID, np.linspace(0, 100, 401)
    else:
        specs, x = OUT, Y
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for name, spec in specs.items():
        ax.plot(x, membership(spec, x), linewidth=2.2, label=name)
    ax.set_ylim(-.03, 1.05)
    ax.set_xlabel(variable_choice); ax.set_ylabel("Membership degree")
    ax.grid(alpha=.2); ax.legend(frameon=False, ncol=len(specs))
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("These curves use the parameters documented in the project report.")

with tab3:
    st.markdown("### Rules activated by the current readings")
    if fired:
        rule_rows = []
        for m, t, h, e, strength in sorted(fired, key=lambda r: r[4], reverse=True):
            rule_rows.append({
                "IF moisture is": m,
                "AND temperature is": t,
                "AND humidity is": h,
                "THEN duration is": e,
                "Firing strength": strength,
            })
        st.dataframe(pd.DataFrame(rule_rows).style.format({"Firing strength": "{:.3f}"}), hide_index=True, use_container_width=True)
    else:
        st.warning("No rules fired for these values. Check the fuzzy membership definitions.")
    st.caption("The full rule base contains 27 combinations. Only rules with non-zero firing strength are listed here.")

st.write("")
st.markdown("### About this prototype")
a, b, c = st.columns(3)
with a:
    st.markdown("**Decision method**")
    st.write("Mamdani fuzzy inference with min–max aggregation and centroid defuzzification.")
with b:
    st.markdown("**Input ranges**")
    st.write("Moisture 0–100%, temperature 0–50°C, humidity 0–100%.")
with c:
    st.markdown("**Validation status**")
    st.write("The report's experiments use synthetic weather and a simulated soil-water balance, not live field sensors.")

st.warning("Academic prototype only: this app does not read physical sensors or control a real valve. Validate recommendations with appropriate agricultural guidance before real-world use.")
st.markdown("---")
st.caption("Smart Irrigation Recommendation System · Soft Computing Project · Built with Python and Streamlit")
