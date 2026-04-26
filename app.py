import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/spotify_predictor_pipeline_TUNED.pkl"
DATA_PATH = "dataset_cleaned.csv"

# Keep dropdown values aligned with training-time super_genre groups.
SUPER_GENRES = [
    "Alternative",
    "Ambient",
    "Country",
    "Jazz/Blues",
    "Rock",
    "Acoustic",
    "Electronic",
    "Pop",
    "R&B/Soul",
    "Latin",
    "Metal",
    "Hip-Hop",
    "Classical",
    "World",
    "Other",
]


def popularity_tier(score: float) -> str:
    if score <= 33:
        return "Flop"
    if score <= 66:
        return "Average"
    return "Hit"


def ensure_duration_min(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "duration_min" not in dataframe.columns:
        dataframe = dataframe.copy()
        dataframe["duration_min"] = dataframe["duration_ms"] / 60000
    return dataframe


def compute_metrics(model, dataframe: pd.DataFrame) -> dict:
    dataframe = ensure_duration_min(dataframe)
    features = [
        "danceability",
        "energy",
        "loudness",
        "valence",
        "tempo",
        "acousticness",
        "explicit",
        "duration_min",
        "super_genre",
    ]

    X = dataframe[features]
    y = dataframe["popularity"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = np.clip(model.predict(X_test), 0, 100)

    y_test_tier = y_test.apply(popularity_tier)
    y_pred_tier = pd.Series(y_pred).apply(popularity_tier)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "tier_accuracy": accuracy_score(y_test_tier, y_pred_tier),
    }


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def build_input_row(
    danceability: float,
    energy: float,
    loudness: float,
    valence: float,
    tempo: float,
    acousticness: float,
    explicit: int,
    duration_min: float,
    super_genre: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "danceability": danceability,
                "energy": energy,
                "loudness": loudness,
                "valence": valence,
                "tempo": tempo,
                "acousticness": acousticness,
                "explicit": explicit,
                "duration_min": duration_min,
                "super_genre": super_genre,
            }
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

    st.title("Spotify Popularity Predictor")
    st.caption("Predict track popularity score and tier (Flop / Average / Hit)")

    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model from '{MODEL_PATH}'.")
        st.exception(exc)
        return

    with st.expander("Model Performance (from dataset_cleaned.csv)"):
        st.caption("Uses an 80/20 holdout split with random_state=42 to mirror your notebook workflow.")
        if st.button("Compute Performance Metrics"):
            try:
                eval_df = pd.read_csv(DATA_PATH)
                metrics = compute_metrics(model, eval_df)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{metrics['mae']:.2f}")
                m2.metric("RMSE", f"{metrics['rmse']:.2f}")
                m3.metric("R²", f"{metrics['r2']:.4f}")
                m4.metric("Tier Accuracy", f"{metrics['tier_accuracy'] * 100:.2f}%")
            except Exception as exc:
                st.error("Could not compute model metrics.")
                st.exception(exc)

    col1, col2, col3 = st.columns(3)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.60, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.60, 0.01)
        valence = st.slider("Valence", 0.0, 1.0, 0.50, 0.01)

    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.20, 0.01)
        tempo = st.slider("Tempo (BPM)", 40.0, 240.0, 120.0, 0.1)
        loudness = st.slider("Loudness (dB)", -60.0, 5.0, -8.0, 0.1)

    with col3:
        duration_min = st.slider("Duration (minutes)", 0.5, 12.0, 3.5, 0.1)
        explicit_label = st.selectbox("Explicit content", ["No", "Yes"], index=0)
        super_genre = st.selectbox("Super Genre", SUPER_GENRES, index=4)

    explicit = 1 if explicit_label == "Yes" else 0

    if st.button("Predict Popularity", type="primary"):
        input_df = build_input_row(
            danceability=danceability,
            energy=energy,
            loudness=loudness,
            valence=valence,
            tempo=tempo,
            acousticness=acousticness,
            explicit=explicit,
            duration_min=duration_min,
            super_genre=super_genre,
        )

        raw_score = float(model.predict(input_df)[0])
        clipped_score = float(np.clip(raw_score, 0, 100))
        tier = popularity_tier(clipped_score)

        st.subheader("Prediction")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Predicted Popularity Score", f"{clipped_score:.2f} / 100")
        metric_col2.metric("Predicted Tier", tier)

        st.progress(clipped_score / 100.0)

        with st.expander("Model Input Sent"):
            st.dataframe(input_df)

    st.divider()
    st.caption("Model file: models/spotify_predictor_pipeline_TUNED.pkl")


if __name__ == "__main__":
    main()
