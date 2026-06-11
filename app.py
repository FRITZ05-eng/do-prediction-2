import os
import io
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
import joblib
from scipy import stats                              # used for trend detection
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image as _PIL_Image

st.title("My App")

DRIVE_FOLDER = "/content/drive/MyDrive/DO_prediction_APP"
if os.path.exists(DRIVE_FOLDER):
    os.chdir(DRIVE_FOLDER)

# ── optional dependency: streamlit-option-menu for a nicer sidebar nav ──────
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# ── optional dependency: TensorFlow for CNN / LSTM models ───────────────────
# We check here (top of file) so every function below can read HAS_TENSORFLOW.
# We catch ALL exceptions (not just ImportError) because TensorFlow can also
# fail with OSError (missing DLLs on Windows), RuntimeError (GPU init issues),
# or other errors that still mean it's effectively unavailable.
try:
    import tensorflow as _tf          # noqa: F401  (imported for side-effects)
    # Extra check: make sure the package is actually functional, not just importable
    _tf_version = _tf.__version__
    HAS_TENSORFLOW = True
    TF_VERSION = _tf_version
except Exception:
    HAS_TENSORFLOW = False
    TF_VERSION = None


# =============================================================================
# CONSTANTS & PATHS
# =============================================================================

MODELS_DIR   = "models"                    # folder that holds .pkl / .keras files
DATA_PATH    = "cleaned_dataset.csv"       # pre-cleaned lake dataset
METRICS_PATH = "model_metrics.csv"         # pre-computed model performance table
BANNER_PATH  = "taal_banner.jpg"           # optional hero image (jpg variant)
LOGO_PATH    = "taal logo.png"             # optional sidebar logo

# Input features expected by every model (order matters for the scaler)
FEATURES = ["Water_Temperature", "pH", "Ammonia", "Nitrate", "Phosphate"]

# The column we are trying to predict
TARGET = "DO"

# Colour palette used consistently for each model throughout the dashboard
PALETTE = {
    "Decision Tree": "#4C72B0",
    "Random Forest": "#DD8452",
    "SVR":           "#55A868",
    "CNN":           "#C44E52",
    "LSTM":          "#8172B2",
}

# Models that require TensorFlow / Keras (used to hide them when TF is absent)
KERAS_MODELS = {"CNN", "LSTM"}

# Ecological DO thresholds (mg/L) used for status classification
DO_CRITICAL = 5.0   # below this → aquatic life is stressed
DO_LOW      = 6.0   # below this → caution zone


# =============================================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# =============================================================================

# Load favicon — fall back to emoji if no logo file is present
if os.path.exists(LOGO_PATH):
    _favicon = _PIL_Image.open(LOGO_PATH)
elif os.path.exists("taal_logo.png"):
    _favicon = _PIL_Image.open("taal_logo.png")
else:
    _favicon = "💧"

st.set_page_config(
    page_title="Taal Lake Water Quality",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# GLOBAL CSS
# =============================================================================
# All CSS lives in one block so it's easy to maintain.
# NOTE: we removed the deprecated .css-1lcbmhc / .css-17lntkn selectors
# (internal Streamlit hash-class names that break across versions).

st.markdown("""
<style>
/* ── Hide radio button circles inside sidebar (nav uses option_menu or
       custom buttons instead) ── */
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] [data-testid="stRadioGroup"],
[data-testid="stSidebar"] input[type="radio"],
[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}

/* ══════════════════════════════════════════
   SIDEBAR — base styling
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e8edf2 !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.06) !important;
}

/* Reset all sidebar text to dark so it's readable on white background */
[data-testid="stSidebar"] * { color: #111827 !important; }

/* Sidebar horizontal dividers */
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid #e8edf2 !important;
    margin: 10px 0 !important;
}

/* Labels for multiselect and radio widgets */
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stRadio label {
    color: #374151 !important;
    font-weight: 600;
}

/* Markdown <hr> inside sidebar */
[data-testid="stSidebar"] .stMarkdown hr {
    border-top: 1px solid #e8edf2 !important;
}

/* option_menu container */
[data-testid="stSidebar"] nav {
    background-color: transparent !important;
}

/* ── Nav link — unselected state ── */
[data-testid="stSidebar"] .nav-link {
    font-size: 16px !important;
    font-weight: 500 !important;
    color: #374151 !important;
    background: transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin: 5px !important;
    border: 1px solid transparent !important;
    transition: background 0.2s ease, color 0.15s ease,
                box-shadow 0.2s ease, transform 0.15s ease !important;
    position: relative !important;
}

/* Nav link — hover state */
[data-testid="stSidebar"] .nav-link:hover {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    color: #1d4ed8 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.10) !important;
    transform: translateX(3px) !important;
}
[data-testid="stSidebar"] .nav-link:hover i,
[data-testid="stSidebar"] .nav-link:hover svg { color: #2563EB !important; }

/* Nav link — active / selected state */
[data-testid="stSidebar"] .nav-link-selected {
    background-color: #2563EB !important;
    border: 1px solid #2563EB !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin: 5px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    transform: none !important;
}
[data-testid="stSidebar"] .nav-link-selected,
[data-testid="stSidebar"] .nav-link-selected * { color: #ffffff !important; }

/* Icons — unselected */
[data-testid="stSidebar"] .nav-link i,
[data-testid="stSidebar"] .nav-link svg {
    color: #2563EB !important;
    transition: color 0.15s ease !important;
}
/* Icons — selected */
[data-testid="stSidebar"] .nav-link-selected i,
[data-testid="stSidebar"] .nav-link-selected svg { color: #ffffff !important; }

/* Sidebar title text */
.sidebar-title {
    color: #2563EB !important;
    font-weight: 800 !important;
    font-size: 20px !important;
    margin-top: 4px;
    margin-bottom: 0;
    letter-spacing: -0.01em;
}

/* Small uppercase section label (e.g. "Date Range") */
.sidebar-section-label {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0 6px;
    margin-bottom: 4px;
}

/* Checkbox label text */
[data-testid="stSidebar"] .stCheckbox label span {
    color: #374151 !important;
    font-size: 14px !important;
}

/* ══════════════════════════════════════════
   Fallback nav buttons (no option_menu)
══════════════════════════════════════════ */
div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100% !important;
    padding: 11px 16px !important;
    margin: 3px 0 !important;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: #374151 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    justify-content: flex-start !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
    color: #1d4ed8 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.12) !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus {
    background: #2563EB !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.40) !important;
}

/* ══════════════════════════════════════════
   MAIN CONTENT COMPONENTS
══════════════════════════════════════════ */

/* KPI / metric card used on the Overview page */
.metric-card {
    background-color: white;
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e8edf2;
    margin-bottom: 8px;
}
.metric-card .label {
    color: #6b7280;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    color: #111827;
    font-size: 28px;
    font-weight: 700;
}

/* Page section header (bold, slightly larger than body) */
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Horizontal radio group outside the sidebar (e.g. data display toggle) */
[data-testid="stRadio"]:not([data-testid="stSidebar"] *) > div {
    flex-direction: row !important;
    gap: 20px !important;
}

/* Prediction result box (large number + label card) */
.pred-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.10);
    border: 1px solid #e8edf2;
    text-align: center;
}
.pred-box .pred-value {
    font-size: 48px;
    font-weight: 800;
    color: #1a3c6e;
}
.pred-box .pred-label { font-size: 16px; color: #6b7280; margin-top: 6px; }

/* Primary action buttons */
.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #1d4ed8;
    transform: scale(1.03);
}

/* Reduce top padding of main content area */
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED DATA & MODEL LOADERS
# =============================================================================
# @st.cache_resource — keeps a single object in memory across reruns (models).
# @st.cache_data    — caches serialisable data (DataFrames) per argument set.

@st.cache_resource
def load_scaler():
    """Load the pre-fitted StandardScaler used to normalise model inputs (X)."""
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))


@st.cache_resource
def load_scaler_y():
    """
    Load the pre-fitted StandardScaler used to normalise the target (DO).

    The models in this package were trained on a scaled target variable, so
    every raw model output is in standardised DO units and must be converted
    back to mg/L using inverse_transform before being shown to the user.

    Returns None if scaler_y.pkl does not exist (backwards-compatibility with
    older model packages that trained on un-scaled DO).
    """
    path = os.path.join(MODELS_DIR, "scaler_y.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_sklearn_model(name: str):
    """
    Load a scikit-learn model (.pkl) by display name.
    The file name convention is: model_<name_lower_snake>.pkl
    e.g. "Random Forest" → models/model_random_forest.pkl
    """
    fname = f"model_{name.lower().replace(' ', '_')}.pkl"
    return joblib.load(os.path.join(MODELS_DIR, fname))


@st.cache_resource
def load_keras_model(name: str):
    """
    Load a Keras model (.keras) by short name ("cnn" or "lstm").
    Raises a clear RuntimeError if TensorFlow is not installed so the
    caller can surface a friendly message rather than a traceback.
    """
    if not HAS_TENSORFLOW:
        raise RuntimeError(
            "TensorFlow is not installed. CNN and LSTM models are unavailable.\n"
            "Run:  pip install tensorflow"
        )
    from tensorflow.keras.models import load_model as km
    return km(os.path.join(MODELS_DIR, f"model_{name.lower()}.keras"))


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and return the cleaned lake dataset as a DataFrame.
    Stops the app with a friendly error message if the file is missing,
    rather than letting Python raise an unhandled FileNotFoundError.
    """
    if not os.path.exists(DATA_PATH):
        st.error(
            f"**Dataset not found:** `{DATA_PATH}`\n\n"
            "Please make sure the cleaned dataset CSV is in the same folder as app.py."
        )
        st.stop()   # halt execution — nothing else can run without data
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """Load the pre-computed model performance metrics table."""
    return pd.read_csv(METRICS_PATH)


# =============================================================================
# PURE UTILITY / HELPER FUNCTIONS
# =============================================================================
# These functions are independent of Streamlit — they take plain data and
# return plain data, making them easy to unit-test in isolation.

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.
    eps prevents division-by-zero when y_true contains zeros.
    Returns a percentage (e.g. 4.2 means 4.2 % error).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_live_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, R², and MAPE from arrays of true and predicted values.
    Returns a dict so callers can pick whichever metrics they need.

    Used to recalculate metrics on-the-fly (e.g. after a batch upload)
    rather than relying solely on the pre-saved model_metrics.csv.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mp   = mape(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE (%)": mp}


def classify_do_status(do_value: float) -> tuple[str, str, str]:
    """
    Classify a dissolved oxygen reading into one of three status categories.

    Returns a tuple of (label, hex_colour, description) so the same
    logic can be reused across the Predictions page, Overview KPIs, and
    any future pages without duplicating the threshold comparisons.

    Thresholds (mg/L):
      < DO_CRITICAL (5.0) → CRITICAL
      < DO_LOW      (6.0) → LOW
      ≥ DO_LOW            → NORMAL
    """
    if do_value < DO_CRITICAL:
        return ("CRITICAL", "#c0392b", "⚠️ CRITICAL — Below 5 mg/L threshold")
    elif do_value < DO_LOW:
        return ("LOW", "#e67e22", "⚠️ LOW — Caution advised")
    else:
        return ("NORMAL", "#27ae60", "✅ ADEQUATE — Safe for aquatic life")


def validate_inputs(
    water_temp: float,
    ph: float,
    ammonia: float,
    nitrate: float,
    phosphate: float,
) -> list[str]:
    """
    Check sensor readings for ecologically implausible combinations.
    Returns a list of warning strings (empty list = all values look fine).

    These are soft warnings — the prediction still runs, but the user is
    informed so they can verify their sensor readings before acting on results.
    """
    warnings = []

    # pH outside the range typical for Taal Lake (slightly alkaline volcanic lake)
    if ph < 5.5:
        warnings.append(f"pH {ph:.2f} is unusually acidic for Taal Lake (typical: 7–9).")
    if ph > 9.5:
        warnings.append(f"pH {ph:.2f} is unusually alkaline — verify sensor calibration.")

    # Very high ammonia indicates heavy organic pollution or sensor error
    if ammonia > 3.0:
        warnings.append(
            f"Ammonia {ammonia:.2f} mg/L is very high — may indicate heavy organic load."
        )

    # High temperature reduces DO saturation capacity
    if water_temp > 35.0:
        warnings.append(
            f"Water temperature {water_temp:.1f}°C is elevated — expect reduced DO capacity."
        )

    # Extreme phosphate is unusual without significant agricultural runoff
    if phosphate > 3.0:
        warnings.append(
            f"Phosphate {phosphate:.2f} mg/L is very high — possible runoff event."
        )

    return warnings


def get_trend(series: pd.Series) -> tuple[str, float]:
    """
    Fit a linear regression to a time-ordered numeric Series and classify
    the slope direction.

    Returns (trend_label, slope) where trend_label is one of:
      "📈 Improving", "📉 Declining", "➡️ Stable"

    'Improving' and 'Declining' are relative to DO — higher DO is better.
    For non-DO parameters the raw slope sign is still informative.
    Threshold: |slope| < 0.005 per unit index is considered stable.
    """
    clean = series.dropna()
    if len(clean) < 3:
        return "➡️ Stable", 0.0

    x = np.arange(len(clean))
    slope, _, _, _, _ = stats.linregress(x, clean.values)

    if abs(slope) < 0.005:
        label = "➡️ Stable"
    elif slope > 0:
        label = "📈 Improving"
    else:
        label = "📉 Declining"

    return label, float(slope)


def detect_anomalies(df: pd.DataFrame, column: str, threshold: float = 2.5) -> pd.Series:
    """
    Return a boolean mask (True = anomaly) for rows where `column` deviates
    more than `threshold` standard deviations from the mean.

    Uses the Z-score method — fast, interpretable, and appropriate for
    roughly normal environmental data.
    """
    col_data = df[column].dropna()
    mean = col_data.mean()
    std  = col_data.std()
    if std == 0:
        return pd.Series(False, index=df.index)   # no variance → no anomalies
    z_scores = (df[column] - mean) / std
    return z_scores.abs() > threshold


def compare_models(inputs: list[float]) -> pd.DataFrame:
    """
    Run all available models on the same input vector and return a DataFrame
    ranked by predicted DO value.

    This gives users a side-by-side view of model agreement (or disagreement)
    without having to manually switch the selectbox six times.

    Columns: Model | Predicted_DO | Status | Colour
    Skips Keras models when TensorFlow is not installed.
    """
    results = []
    for model_name in PALETTE.keys():
        # Skip deep learning models when TF isn't available
        if model_name in KERAS_MODELS and not HAS_TENSORFLOW:
            continue
        try:
            do_val = predict(model_name, inputs)
            label, colour, _ = classify_do_status(do_val)
            results.append({
                "Model":        model_name,
                "Predicted_DO": round(do_val, 3),
                "Status":       label,
                "Colour":       colour,
            })
        except Exception:
            # If a specific model file is missing, skip it gracefully
            pass

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out = df_out.sort_values("Predicted_DO", ascending=False).reset_index(drop=True)
    return df_out


def suggest_remediation(do_value: float, ph: float, ammonia: float) -> list[str]:
    """
    Return plain-language remediation suggestions based on current readings.

    Designed for fish farm operators and environmental managers who need
    actionable next steps, not just numbers.
    Returns a list of suggestion strings (empty = no action needed).
    """
    suggestions = []

    if do_value < DO_CRITICAL:
        suggestions.append(
            "🔧 **Aeration:** DO is critically low. Deploy aerators or paddlewheels "
            "immediately to increase oxygen transfer."
        )
    if do_value < DO_LOW:
        suggestions.append(
            "📉 **Reduce stocking density** temporarily and avoid feeding heavily "
            "until DO recovers above 6 mg/L."
        )
    if ammonia > 1.0:
        suggestions.append(
            "🧪 **Ammonia is elevated.** Check for overfeeding or dead biomass. "
            "Consider a partial water exchange or biological filter."
        )
    if ph < 6.5:
        suggestions.append(
            "🌿 **Low pH** — consider lime application to buffer acidity, "
            "especially during algae die-off periods."
        )
    if ph > 9.0:
        suggestions.append(
            "🌞 **High pH** may indicate algal bloom. Monitor turbidity and "
            "consider algaecide treatment if bloom is confirmed."
        )
    if not suggestions:
        suggestions.append("✅ Water quality parameters appear within acceptable ranges.")

    return suggestions


def log_prediction(
    model_name: str,
    inputs: list[float],
    do_result: float,
) -> None:
    """
    Append a prediction record to st.session_state so the user can review
    their prediction history within the current session.

    Each record is a dict with timestamp, model, inputs, and result.
    History is cleared when the browser tab is closed / session resets.
    """
    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = []

    record = {
        "Time":              datetime.datetime.now().strftime("%H:%M:%S"),
        "Model":             model_name,
        "Water_Temperature": inputs[0],
        "pH":                inputs[1],
        "Ammonia":           inputs[2],
        "Nitrate":           inputs[3],
        "Phosphate":         inputs[4],
        "Predicted_DO":      round(do_result, 3),
        "Status":            classify_do_status(do_result)[0],
    }
    st.session_state.prediction_log.append(record)


def load_data_with_validation() -> pd.DataFrame:
    """
    Load the dataset and validate that all expected columns are present
    and have the correct data types.

    Stops the app early with a descriptive error rather than letting a
    confusing KeyError surface deep inside a plotting function.
    """
    df = load_data()   # uses the cached loader above

    # Check that all model features exist in the dataset
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        st.error(
            f"**Schema error:** The following expected columns are missing from "
            f"`{DATA_PATH}`: `{missing_cols}`\n\n"
            "Please verify the dataset or update the FEATURES / TARGET constants."
        )
        st.stop()

    # Warn if any feature column is entirely null
    for col in FEATURES + [TARGET]:
        if df[col].isnull().all():
            st.warning(f"Column `{col}` contains only null values — check the dataset.")

    return df


# =============================================================================
# MODEL INFERENCE
# =============================================================================

def predict(model_name: str, inputs: list[float]) -> float:
    """
    Scale the input vector, run it through the selected model, and
    inverse-transform the raw output back to mg/L.

    Parameters
    ----------
    model_name : str
        One of the keys in PALETTE (e.g. "Random Forest", "LSTM").
    inputs : list of float
        Raw sensor values in FEATURES order:
        [Water_Temperature, pH, Ammonia, Nitrate, Phosphate]

    Returns
    -------
    float
        Predicted dissolved oxygen value in mg/L (actual scale, not
        standardised).

    Raises
    ------
    RuntimeError
        If a Keras model is requested but TensorFlow is not installed.
    ValueError
        If an unrecognised model name is passed.

    Notes
    -----
    The models in this package were trained with a scaled target variable
    (StandardScaler applied to DO before fitting). Raw model outputs are
    therefore in standardised units; scaler_y.inverse_transform() converts
    them back to mg/L before returning.
    """
    # Guard: refuse Keras models when TF is absent
    if model_name in KERAS_MODELS and not HAS_TENSORFLOW:
        raise RuntimeError(
            f"**{model_name}** requires TensorFlow which is not installed.\n\n"
            "Please select **Decision Tree**, **Random Forest**, or **SVR** — "
            "or install TensorFlow with:  `pip install tensorflow`"
        )

    # Scale inputs using the pre-fitted feature scaler
    scaler   = load_scaler()
    scaler_y = load_scaler_y()          # may be None for older model packages
    Xs = scaler.transform(np.array([inputs]))   # shape: (1, n_features)

    def _inverse(raw: float) -> float:
        """Convert a raw (possibly scaled) prediction back to mg/L."""
        if scaler_y is not None:
            return float(scaler_y.inverse_transform([[raw]])[0][0])
        return float(raw)

    if model_name in ("Decision Tree", "Random Forest", "SVR"):
        # Standard sklearn .predict() → returns a 1-element array
        raw = float(load_sklearn_model(model_name).predict(Xs))
        return _inverse(raw)

    elif model_name == "CNN":
        # CNN expects shape (batch, n_features) — same as sklearn
        raw = float(load_keras_model("cnn").predict(Xs, verbose=0).flatten())
        return _inverse(raw)

    elif model_name == "LSTM":
        # LSTM expects shape (batch, timesteps, features); we use 1 timestep
        X3  = Xs.reshape(1, 1, Xs.shape[1])
        raw = float(load_keras_model("lstm").predict(X3, verbose=0).flatten())
        return _inverse(raw)

    else:
        # This should never be reached if PALETTE is kept in sync with the
        # model files — but a clear error beats a silent None return.
        raise ValueError(f"Unknown model name: '{model_name}'")


# =============================================================================
# UI HELPERS
# =============================================================================

def show_banner() -> None:
    """
    Display the Taal Lake banner image at the top of each page.
    Falls back to a CSS gradient card if no image file is found,
    so the app looks polished even without asset files.
    """
    if os.path.exists(BANNER_PATH):
        st.image(BANNER_PATH, use_container_width=True)
    elif os.path.exists("taal_banner.png"):
        st.image("taal_banner.png", use_container_width=True)
    else:
        # Gradient fallback banner
        st.markdown(
            "<div style='background:linear-gradient(135deg,#1a3c6e,#2e6da4);"
            "border-radius:12px;padding:32px;text-align:center;margin-bottom:16px'>"
            "<span style='color:white;font-size:28px;font-weight:800'>"
            "💧 Taal Lake Water Quality</span><br>"
            "<span style='color:#cce;font-size:14px'>Prediction Dashboard</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def section_header(title: str) -> None:
    """Render a styled page section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    # ── Logo ────────────────────────────────────────────────────────────────
    _logo_file = (
        LOGO_PATH if os.path.exists(LOGO_PATH)
        else ("taal_logo.png" if os.path.exists("taal_logo.png") else None)
    )
    if _logo_file:
        st.image(_logo_file, width=300)
    else:
        st.markdown("## 💧")

    # ── App title ────────────────────────────────────────────────────────────
    st.markdown(
        "<p class='sidebar-title'>Dissolved Oxygen Prediction</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:8px 0; border-color:#C97B22;'>", unsafe_allow_html=True)

    # ── Navigation ──────────────────────────────────────────────────────────
    page_options = [
        "Overview",
        "Time Series",
        "Feature Analysis",
        "Model Metrics",
        "Predictions",
        "Model Visualizations",
    ]

    if HAS_OPTION_MENU:
        # Preferred: uses streamlit-option-menu for icon support
        page = option_menu(
            menu_title=None,
            options=page_options,
            icons=["house-fill", "graph-up", "search",
                   "bar-chart-line", "magic", "display"],
            default_index=0,
            key="main_nav",
            styles={
                "container":       {"padding": "4px 0", "background-color": "transparent"},
                "icon":            {"color": "#2563EB", "font-size": "18px"},
                "nav-link": {
                    "font-size": "15px", "font-weight": "500",
                    "text-align": "left", "margin": "3px 0",
                    "padding": "11px 16px", "border-radius": "12px",
                    "color": "#374151", "background-color": "transparent",
                    "border": "1px solid transparent", "transition": "all 0.2s ease",
                },
                "nav-link-selected": {
                    "background-color": "#2563EB", "color": "#ffffff",
                    "font-weight": "700", "border-radius": "12px",
                    "border": "1px solid #1d4ed8",
                    "box-shadow": "0 4px 16px rgba(37,99,235,0.40)",
                },
            },
        )
    else:
        # Fallback: plain Streamlit buttons styled to look like nav items
        # Session state is used so the selected page persists across reruns
        if "nav_page" not in st.session_state:
            st.session_state.nav_page = "Overview"

        for opt in page_options:
            if st.button(opt, key=f"nav_{opt}", use_container_width=True):
                st.session_state.nav_page = opt
                st.rerun()

        page = st.session_state.nav_page

    st.markdown("<hr style='margin:8px 0; border-color:#C97B22;'>", unsafe_allow_html=True)

    # ── Date range filter ────────────────────────────────────────────────────
    # We derive min/max from the actual data rather than hardcoding years,
    # so the slider stays correct if the dataset is updated.
    st.markdown(
        "<div class='sidebar-section-label'>📅 Date Range</div>",
        unsafe_allow_html=True,
    )

    # Load data early just to get the year bounds for the slider
    _df_for_bounds = load_data()
    if "Year" in _df_for_bounds.columns:
        _year_min = int(_df_for_bounds["Year"].min())
        _year_max = int(_df_for_bounds["Year"].max())
    else:
        _year_min, _year_max = 2013, 2023   # safe fallback

    use_full_range = st.checkbox("Use Full Date Range", value=True)

    if not use_full_range:
        # Persist slider value in session state so re-checking the box resets it
        year_range = st.slider(
            "Year Range",
            _year_min, _year_max,
            st.session_state.get("year_range", (_year_min, _year_max)),
        )
        st.session_state.year_range = year_range
    else:
        year_range = (_year_min, _year_max)
        # Clear any cached slider value so unchecking starts fresh
        if "year_range" in st.session_state:
            del st.session_state.year_range


# =============================================================================
# LOAD & FILTER DATA
# =============================================================================

# Use the validated loader so schema errors surface before any page renders
df_full    = load_data_with_validation()
metrics_df = load_metrics() if os.path.exists(METRICS_PATH) else None

# Apply the year filter selected in the sidebar
if "Year" in df_full.columns:
    df = df_full[
        (df_full["Year"] >= year_range[0]) &
        (df_full["Year"] <= year_range[1])
    ].copy()
else:
    df = df_full.copy()


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

if page == "Overview":
    show_banner()
    section_header("🏞️ Water Quality Overview")

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    total_records = len(df)
    n_params      = df.shape[1]
    start_date    = f"{int(df['Year'].min())}-01-01" if "Year" in df else "—"
    end_date      = f"{int(df['Year'].max())}-12-01" if "Year" in df else "—"

    # DO trend direction across the filtered date range
    do_trend_label, do_slope = get_trend(df[TARGET]) if TARGET in df.columns else ("—", 0.0)

    for col, label, value in zip(
        [c1, c2, c3, c4, c5],
        ["Total Records", "Parameters", "Start Date", "End Date", "DO Trend"],
        [f"{total_records:,}", str(n_params), start_date, end_date, do_trend_label],
    ):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Raw data explorer ────────────────────────────────────────────────────
    section_header("📋 Complete Raw Data")

    display_mode = st.radio(
        "Data Display Options:",
        ["View All Data", "Filter by Column", "Search Records"],
        horizontal=True,
    )

    if display_mode == "View All Data":
        st.dataframe(df, width='stretch', height=380)

    elif display_mode == "Filter by Column":
        col_filter  = st.selectbox("Select column to filter", df.columns.tolist())
        unique_vals = df[col_filter].dropna().unique()
        selected    = st.multiselect(
            f"Filter values for '{col_filter}'",
            sorted(unique_vals.tolist()),
            default=sorted(unique_vals.tolist())[:5],
        )
        filtered = df[df[col_filter].isin(selected)]
        st.dataframe(filtered, width='stretch', height=340)
        st.caption(f"Showing {len(filtered):,} of {len(df):,} records")

    else:   # Search Records
        search_term = st.text_input("Search (applies to all text columns)")
        if search_term:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            st.dataframe(df[mask], width='stretch', height=340)
            st.caption(f"Found {mask.sum():,} matching records")
        else:
            st.dataframe(df.head(50), width='stretch', height=340)

    # ── Descriptive statistics ───────────────────────────────────────────────
    section_header("📊 Descriptive Statistics")
    avail = [c for c in FEATURES + [TARGET] if c in df.columns]
    st.dataframe(df[avail].describe().round(3), width='stretch')

    # ── Missing values report ────────────────────────────────────────────────
    section_header("🔍 Missing Values")
    miss = df[avail].isnull().sum().rename("Missing Count").to_frame()
    miss["% Missing"] = (miss["Missing Count"] / len(df) * 100).round(2)
    st.dataframe(miss, width='stretch')

    # ── Anomaly detection ────────────────────────────────────────────────────
    section_header("⚠️ Anomaly Detection")
    st.markdown(
        "Rows where any feature deviates more than **2.5 standard deviations** "
        "from its mean are flagged as potential anomalies."
    )

    anomaly_mask = pd.Series(False, index=df.index)
    for feat in avail:
        anomaly_mask |= detect_anomalies(df, feat, threshold=2.5)

    anomaly_df = df[anomaly_mask]
    st.info(f"Found **{len(anomaly_df):,}** anomalous records out of {len(df):,} total.")
    if not anomaly_df.empty:
        st.dataframe(anomaly_df[avail].head(100), width='stretch')


# =============================================================================
# PAGE 2 — TIME SERIES
# =============================================================================

elif page == "Time Series":
    show_banner()
    section_header("📈 Time Series Analysis")

    # Build a continuous numeric time index (fractional year) for plotting
    month_order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    ]
    df["Month_num"]  = pd.Categorical(
        df["Month"], categories=month_order, ordered=True
    ).codes + 1
    df["Time_index"] = df["Year"] + (df["Month_num"] - 1) / 12.0

    # ── DO over time ─────────────────────────────────────────────────────────
    st.subheader("Dissolved Oxygen Over Time")

    ts = df.groupby("Time_index")[TARGET].agg(["mean", "min", "max"]).reset_index()
    trend_label, slope = get_trend(ts["mean"])
    st.caption(f"Overall DO trend: **{trend_label}** (slope = {slope:+.4f} mg/L per month)")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(ts["Time_index"], ts["min"], ts["max"],
                    alpha=0.15, color="#1a3c6e", label="Min–Max range")
    ax.plot(ts["Time_index"], ts["mean"], color="#1a3c6e",
            linewidth=2, label="Monthly mean DO")
    ax.axhline(DO_CRITICAL, color="red", linestyle="--", linewidth=1.2,
               label=f"{DO_CRITICAL} mg/L critical threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("DO (mg/L)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("DO Over Time (All Sites)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Multi-parameter time series ──────────────────────────────────────────
    st.subheader("Multi-Parameter Time Series")
    avail_feat  = [c for c in FEATURES if c in df.columns]
    param_choice = st.multiselect(
        "Select parameters to plot", avail_feat, default=avail_feat[:2]
    )
    if param_choice:
        fig, ax = plt.subplots(figsize=(14, 4))
        cmap = plt.cm.tab10
        for i, feat in enumerate(param_choice):
            ts2 = df.groupby("Time_index")[feat].mean()
            ax.plot(ts2.index, ts2.values, linewidth=1.8, label=feat, color=cmap(i))
        ax.set_xlabel("Year")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_title("Selected Parameters Over Time", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Seasonal average ─────────────────────────────────────────────────────
    st.subheader("Seasonal Monthly Average DO")
    if "Month_num" in df.columns:
        monthly = df.groupby("Month_num")[TARGET].mean().reset_index()
        monthly["Month"] = [month_order[int(m) - 1][:3] for m in monthly["Month_num"]]

        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(
            monthly["Month"], monthly[TARGET],
            color="#1a3c6e", alpha=0.85, edgecolor="white",
        )
        ax.axhline(DO_CRITICAL, color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel("Mean DO (mg/L)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Average DO by Month (All Years)", fontweight="bold")
        for bar, v in zip(bars, monthly[TARGET]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=8,
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# =============================================================================
# PAGE 3 — FEATURE ANALYSIS
# =============================================================================

elif page == "Feature Analysis":
    show_banner()
    section_header("🔬 Feature Analysis")

    avail  = [c for c in FEATURES + [TARGET] if c in df.columns]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    # ── Distributions ────────────────────────────────────────────────────────
    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(avail):
        axes[i].hist(
            df[col].dropna(), bins=40, color=colors[i],
            edgecolor="white", linewidth=0.5, alpha=0.88,
        )
        median_val = df[col].median()
        axes[i].axvline(
            median_val, color="red", linestyle="--", linewidth=1.4,
            label=f"Median={median_val:.2f}",
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(fontsize=8)
        axes[i].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.subheader("Correlation Heatmap")
    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    # mask=upper triangle so we only show the lower half (avoids redundancy)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 10},
        vmin=-1, vmax=1,
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Feature vs DO scatter plots ──────────────────────────────────────────
    st.subheader("Feature vs DO Scatter Plots")
    feat_cols = [c for c in FEATURES if c in df.columns]
    fig, axes = plt.subplots(1, len(feat_cols), figsize=(18, 4))
    for i, feat in enumerate(feat_cols):
        axes[i].scatter(
            df[feat], df[TARGET],
            alpha=0.3, s=12, color=colors[i], edgecolors="none",
        )
        axes[i].set_xlabel(feat, fontsize=9)
        axes[i].set_ylabel("DO", fontsize=9)
        axes[i].set_title(f"{feat} vs DO", fontsize=9, fontweight="bold")
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Box plots ────────────────────────────────────────────────────────────
    st.subheader("Box Plot Overview")
    fig, axes = plt.subplots(1, len(avail), figsize=(18, 5))
    for i, col in enumerate(avail):
        bp = axes[i].boxplot(
            df[col].dropna(), patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(colors[i])
        bp["boxes"][0].set_alpha(0.75)
        axes[i].set_title(col, fontsize=9, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =============================================================================
# PAGE 4 — MODEL METRICS
# =============================================================================

elif page == "Model Metrics":
    show_banner()
    section_header("📊 Model Performance Metrics")

    if metrics_df is not None:
        # Highlight best model by RMSE (lower is better)
        best = metrics_df.sort_values("RMSE").iloc[0]
        st.success(
            f"🏆 Best model by RMSE: **{best['Model']}** "
            f"— RMSE={best['RMSE']:.4f} | MAE={best['MAE']:.4f} | R²={best['R²']:.4f}"
        )

        # Per-model R² cards — use model name for colour lookup (not index)
        cols = st.columns(len(metrics_df))
        for col, (_, row) in zip(cols, metrics_df.iterrows()):
            model_colour = PALETTE.get(row["Model"], "#333333")
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='label'>{row['Model']}</div>"
                f"<div class='value' style='font-size:20px;color:{model_colour}'>"
                f"R²={row['R²']:.3f}</div>"
                f"<div style='font-size:12px;color:#888;margin-top:6px'>"
                f"RMSE={row['RMSE']:.3f} | MAE={row['MAE']:.3f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Full metrics table ───────────────────────────────────────────────
        st.subheader("Full Metrics Table")
        st.dataframe(metrics_df.round(4), width='stretch')

        # ── Bar charts for each metric ───────────────────────────────────────
        st.subheader("Metrics Bar Charts")
        metric_cols = [c for c in ["RMSE", "MAE", "R²", "MAPE (%)"] if c in metrics_df.columns]
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(16, 4))
        if len(metric_cols) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_cols):
            # Map bar colours by model name so order in CSV doesn't matter
            bar_colors = [PALETTE.get(m, "#999") for m in metrics_df["Model"]]
            bars = ax.bar(
                metrics_df["Model"], metrics_df[metric],
                color=bar_colors, edgecolor="white", alpha=0.88,
            )
            ax.set_title(metric, fontweight="bold")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.3)
            ax.set_xticklabels(metrics_df["Model"], rotation=20, ha="right", fontsize=9)
            # Value labels above each bar
            for bar, v in zip(bars, metrics_df[metric]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(metrics_df[metric]) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.warning(
            "⚠️ `model_metrics.csv` was not found. "
            "Run the training notebook first to generate it."
        )


# =============================================================================
# PAGE 5 — PREDICTIONS
# =============================================================================

elif page == "Predictions":
    show_banner()
    section_header("🔮 Predict Dissolved Oxygen")

    st.markdown(
        "Enter current water quality sensor readings to get a real-time "
        "dissolved oxygen prediction."
    )

    # ── TensorFlow availability notice ───────────────────────────────────────
    # Shows version when found, or a diagnostic message when not.
    # If you see the warning despite having TF installed, the cause is almost
    # always a mismatch between the Python/venv where TF was pip-installed and
    # the Python that Streamlit is actually using. Solution: activate the correct
    # environment BEFORE running `streamlit run app.py`.
    if HAS_TENSORFLOW:
        st.success(
            f"✅ **TensorFlow {TF_VERSION} detected** — CNN and LSTM models are available.",
            icon=None,
        )
    else:
        st.warning(
            "⚠️ **TensorFlow not detected** — CNN and LSTM models are hidden.\n\n"
            "If TensorFlow IS installed but you still see this, launch the app from "
            "the **same environment** where you installed it:\n\n"
            "```\n"
            "# Windows (venv)\n"
            ".venv\\Scripts\\streamlit run app.py\n\n"
            "# Mac / Linux\n"
            "source .venv/bin/activate && streamlit run app.py\n"
            "```\n\n"
            "You can still use **Decision Tree**, **Random Forest**, and **SVR** below.",
            icon=None,
        )

    # Only expose models whose dependencies are met
    available_models = [m for m in PALETTE.keys() if m not in KERAS_MODELS or HAS_TENSORFLOW]

    # ── Input form ───────────────────────────────────────────────────────────
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            model_choice = st.selectbox("Select Model", available_models)
            water_temp   = st.number_input("Water Temperature (°C)", 20.0, 40.0, 27.0, 0.1)
        with col2:
            ph      = st.number_input("pH", 4.0, 10.0, 8.0, 0.01)
            ammonia = st.number_input("Ammonia NH₃ (mg/L)", 0.0, 5.0, 0.3, 0.01)
        with col3:
            nitrate   = st.number_input("Nitrate NO₃ (mg/L)", 0.0, 30.0, 0.1, 0.01)
            phosphate = st.number_input("Phosphate PO₄ (mg/L)", 0.0, 5.0, 0.05, 0.01)

    # Collect all inputs into a list for reuse across single + compare predictions
    current_inputs = [water_temp, ph, ammonia, nitrate, phosphate]

    # ── Input validation warnings (shown before prediction) ─────────────────
    input_warnings = validate_inputs(water_temp, ph, ammonia, nitrate, phosphate)
    if input_warnings:
        with st.expander("⚠️ Input Warnings — click to expand", expanded=True):
            for w in input_warnings:
                st.warning(w)

    # ── Action buttons ───────────────────────────────────────────────────────
    btn_col1, btn_col2 = st.columns(2)
    predict_btn = btn_col1.button("🔍 Predict DO", type="primary", use_container_width=True)
    compare_btn = btn_col2.button("⚖️ Compare All Models", use_container_width=True)

    # ── Single-model prediction ──────────────────────────────────────────────
    if predict_btn:
        try:
            do_pred = predict(model_choice, current_inputs)
            status_label, card_color, status_msg = classify_do_status(do_pred)

            # Show status banner
            if status_label == "CRITICAL":
                st.error(status_msg)
            elif status_label == "LOW":
                st.warning(status_msg)
            else:
                st.success(status_msg)

            # Result cards
            r1, r2, r3 = st.columns(3)
            r1.markdown(
                f"<div class='pred-box'>"
                f"<div class='pred-label'>Predicted DO</div>"
                f"<div class='pred-value' style='color:{card_color}'>{do_pred:.3f}</div>"
                f"<div class='pred-label'>mg/L</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            r2.markdown(
                f"<div class='pred-box'>"
                f"<div class='pred-label'>Model Used</div>"
                f"<div class='pred-value' style='font-size:24px;color:{PALETTE[model_choice]}'>"
                f"{model_choice}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            r3.markdown(
                f"<div class='pred-box'>"
                f"<div class='pred-label'>Status</div>"
                f"<div class='pred-value' style='font-size:22px;color:{card_color}'>"
                f"{status_label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # DO gauge bar chart
            fig, ax = plt.subplots(figsize=(10, 1.4))
            ax.barh(0, 15, color="#e8ecf0", height=0.6)                        # full range
            ax.barh(0, min(do_pred, 15), color=card_color, height=0.6, alpha=0.85)  # actual
            ax.axvline(DO_CRITICAL, color="#c0392b", linewidth=2, linestyle="--")
            ax.axvline(DO_LOW,      color="#e67e22", linewidth=1.5, linestyle=":")
            ax.set_xlim(0, 15)
            ax.set_yticks([])
            ax.set_xlabel("DO (mg/L)", fontsize=10)
            ax.set_title(f"DO Level Gauge — {do_pred:.3f} mg/L", fontweight="bold")
            ax.text(5.1,  0.32, "5 mg/L\ncritical", va="center", fontsize=7.5, color="#c0392b")
            ax.text(6.1, -0.32, "6 mg/L\nlow",      va="center", fontsize=7.5, color="#e67e22")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Remediation suggestions ──────────────────────────────────────
            st.markdown("---")
            st.subheader("💡 Recommendations")
            for suggestion in suggest_remediation(do_pred, ph, ammonia):
                st.markdown(suggestion)

            # Save to session log
            log_prediction(model_choice, current_inputs, do_pred)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ── Compare all models ───────────────────────────────────────────────────
    if compare_btn:
        st.markdown("---")
        st.subheader("⚖️ All-Model Comparison")

        with st.spinner("Running all models…"):
            comp_df = compare_models(current_inputs)

        if comp_df.empty:
            st.warning("No models could run. Check that model files exist in the `models/` folder.")
        else:
            # Colour-coded table
            st.dataframe(
                comp_df[["Model", "Predicted_DO", "Status"]],
                width='stretch',
            )

            # Bar chart of all predictions
            fig, ax = plt.subplots(figsize=(8, 3))
            bar_colors = [PALETTE.get(m, "#999") for m in comp_df["Model"]]
            bars = ax.bar(comp_df["Model"], comp_df["Predicted_DO"],
                          color=bar_colors, edgecolor="white", alpha=0.88)
            ax.axhline(DO_CRITICAL, color="#c0392b", linestyle="--", linewidth=1.2,
                       label=f"Critical ({DO_CRITICAL} mg/L)")
            ax.axhline(DO_LOW, color="#e67e22", linestyle=":", linewidth=1.2,
                       label=f"Low ({DO_LOW} mg/L)")
            ax.set_ylabel("Predicted DO (mg/L)")
            ax.set_title("Model Comparison — Predicted DO", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, comp_df["Predicted_DO"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{v:.3f}", ha="center", fontsize=9,
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Batch prediction (always visible — not gated behind predict_btn) ─────
    st.markdown("---")
    st.subheader("📁 Batch Prediction")
    st.markdown(
        "Upload a CSV with the same feature columns to predict DO for multiple records at once.\n\n"
        "**Expected columns:** `Water_Temperature`, `pH`, `Ammonia`, `Nitrate`, `Phosphate`"
    )

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], key="batch_upload"
    )

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

        # Normalise common alternative column names
        rename_map = {
            "Surface Temp":      "Water_Temperature",
            "Water Temperature": "Water_Temperature",
        }
        batch_df = batch_df.rename(columns=rename_map)

        avail_f = [c for c in FEATURES if c in batch_df.columns]

        if len(avail_f) == len(FEATURES):
            # Warn if any values were imputed
            nan_counts = batch_df[FEATURES].isnull().sum()
            imputed_cols = nan_counts[nan_counts > 0].index.tolist()
            if imputed_cols:
                st.warning(
                    f"Missing values detected in: `{imputed_cols}`. "
                    "These were filled with column medians before prediction."
                )

            scaler_obj = load_scaler()
            scaler_y   = load_scaler_y()    # needed to convert scaled predictions back to mg/L
            Xb = scaler_obj.transform(
                batch_df[FEATURES].fillna(batch_df[FEATURES].median())
            )

            preds = None
            if model_choice in ("Decision Tree", "Random Forest", "SVR"):
                m     = load_sklearn_model(model_choice)
                preds = m.predict(Xb)
            elif model_choice in KERAS_MODELS:
                if not HAS_TENSORFLOW:
                    st.error("TensorFlow is required for this model. Run: `pip install tensorflow`")
                elif model_choice == "CNN":
                    m     = load_keras_model("cnn")
                    preds = m.predict(Xb, verbose=0).flatten()
                else:
                    m     = load_keras_model("lstm")
                    preds = m.predict(
                        Xb.reshape(Xb.shape[0], 1, Xb.shape[1]), verbose=0
                    ).flatten()

            # Inverse-transform from scaled DO units back to mg/L
            if preds is not None and scaler_y is not None:
                preds = scaler_y.inverse_transform(
                    preds.reshape(-1, 1)
                ).flatten()

            if preds is not None:
                batch_df["Predicted_DO"] = preds.round(3)
                # Add status column using classify_do_status
                batch_df["DO_Status"] = batch_df["Predicted_DO"].apply(
                    lambda v: classify_do_status(v)[0]
                )

                # Show live metrics if the ground-truth DO column is present
                if TARGET in batch_df.columns:
                    st.subheader("📐 Live Batch Metrics")
                    live = compute_live_metrics(
                        batch_df[TARGET].values, batch_df["Predicted_DO"].values
                    )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("RMSE",     f"{live['RMSE']:.4f}")
                    m2.metric("MAE",      f"{live['MAE']:.4f}")
                    m3.metric("R²",       f"{live['R²']:.4f}")
                    m4.metric("MAPE (%)", f"{live['MAPE (%)']:.2f}")

                st.dataframe(batch_df, width='stretch')
                csv_out = batch_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out,
                    "batch_predictions.csv",
                    "text/csv",
                )
        else:
            missing_f = [c for c in FEATURES if c not in batch_df.columns]
            st.warning(
                f"**Missing columns:** `{missing_f}`\n\n"
                f"Expected: `{FEATURES}`\n"
                f"Found: `{batch_df.columns.tolist()}`"
            )

    # ── Prediction history log ───────────────────────────────────────────────
    if st.session_state.get("prediction_log"):
        st.markdown("---")
        st.subheader("🕓 Session Prediction History")
        log_df = pd.DataFrame(st.session_state.prediction_log)
        st.dataframe(log_df, width='stretch')
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_log = []
            st.rerun()


# =============================================================================
# PAGE 6 — MODEL VISUALIZATIONS
# =============================================================================

elif page == "Model Visualizations":
    show_banner()
    section_header("🖼️ Model Evaluation Visualizations")

    # Map display label → (relative file path, stage category)
    plot_files = {
        "Loss Curve — CNN":                ("plots/08_cnn_loss_curve.png",               "Training"),
        "Loss Curve — LSTM":               ("plots/08_lstm_loss_curve.png",              "Training"),
        "Actual vs Predicted (Scatter)":   ("plots/09_scatter_actual_vs_predicted.png",  "Evaluation"),
        "Actual vs Predicted (Line Plot)": ("plots/10_lineplot_actual_vs_predicted.png", "Evaluation"),
        "Residual Plots":                  ("plots/11_residual_plots.png",               "Evaluation"),
        "Model Metrics Bar Chart":         ("plots/07_model_metrics_barchart.png",       "Evaluation"),
        "Feature Importance (RF)":         ("plots/06_feature_importance.png",           "Features"),
        "Histograms":                      ("plots/01_histograms.png",                   "EDA"),
        "Time-Series DO":                  ("plots/02_timeseries_DO.png",                "EDA"),
        "Correlation Heatmap":             ("plots/03_correlation_heatmap.png",          "EDA"),
        "Box Plot Overview":               ("plots/04_boxplot_overview.png",             "EDA"),
        "Pair Plot":                       ("plots/05_pairplot.png",                     "EDA"),
    }

    categories = ["All", "EDA", "Features", "Training", "Evaluation"]
    tab_sel = st.radio("Filter by stage:", categories, horizontal=True)

    # Count how many plots are available vs missing for the selected filter
    filtered_plots = {
        label: (path, cat)
        for label, (path, cat) in plot_files.items()
        if tab_sel == "All" or cat == tab_sel
    }
    found   = sum(1 for _, (p, _) in filtered_plots.items() if os.path.exists(p))
    missing = len(filtered_plots) - found

    # Single informational callout if plots are missing (replaces wall of captions)
    if missing > 0:
        st.info(
            f"**{missing} plot(s) not yet generated** for the '{tab_sel}' stage. "
            "Run the training/EDA notebook to produce them, then refresh this page."
        )

    # Render each available plot in an expandable section
    for label, (rel_path, _) in filtered_plots.items():
        if os.path.exists(rel_path):
            with st.expander(f"📊 {label}", expanded=False):
                st.image(rel_path, use_container_width=True)
