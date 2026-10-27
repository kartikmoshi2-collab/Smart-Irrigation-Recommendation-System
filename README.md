# Smart Irrigation Recommendation System

A Streamlit dashboard for the Soft Computing project. The fuzzy controller follows the Mamdani inference design documented in the project report:
- Inputs: soil moisture (%), air temperature (°C), relative humidity (%)
- Output: recommended irrigation duration (0–60 minutes)
- 27-rule fuzzy base, min operator for AND, max aggregation, centroid defuzzification
- Minimum run-time threshold: 4 minutes

## Run locally in VS Code

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

6. Streamlit will show a local URL, usually `http://localhost:8501`.

## Deploy online (free Streamlit Community Cloud)

1. Create/sign in to GitHub and make a new repository, e.g. `smart-irrigation-dashboard`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Open https://share.streamlit.io and sign in with GitHub.
4. Click **Create app**, select your repository, branch (`main`), and entry file (`app.py`).
5. Click **Deploy**. When it finishes, copy the generated `*.streamlit.app` URL.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Important note

This is a simulation / academic prototype. It does not connect to real sensors or operate a physical irrigation valve. Project report results are based on synthetic weather and should not be presented as field validation.
