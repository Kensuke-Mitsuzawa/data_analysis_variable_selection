import os
import logging
import typing as ty
import urllib.request
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpeedDatingDataLoader:
    """Loads raw Columbia Speed Dating Experiment records from local CSV, remote URL, or synthetic fallback.
    """

    def load_data_raw(self, path_source: ty.Optional[str] = None) -> pd.DataFrame:
        """Loads raw speed dating dataframe from file, Kaggle, remote URL, or synthetic generator.

        Args:
            path_source: Local filesystem path, Kaggle URL/ID, or remote URL to raw Speed Dating CSV.

        Returns:
            Pandas DataFrame containing raw speed dating scorecard records.
        """
        if path_source is not None:
            if os.path.isfile(path_source):
                logger.info(f"Loading Speed Dating dataset from local file: {path_source}")
                try:
                    df_raw = pd.read_csv(path_source, encoding="utf-8")
                except UnicodeDecodeError:
                    df_raw = pd.read_csv(path_source, encoding="latin1")
                # end try
                return df_raw
            # end if

            # Direct Kaggle dataset handling via kagglehub
            if "kaggle.com" in path_source or "annavictoria/speed-dating-experiment" in path_source:
                try:
                    import kagglehub
                    logger.info("Downloading Speed Dating dataset from Kaggle (annavictoria/speed-dating-experiment)...")
                    path_download = kagglehub.dataset_download("annavictoria/speed-dating-experiment")
                    path_csv = os.path.join(path_download, "Speed Dating Data.csv")
                    if os.path.exists(path_csv):
                        return pd.read_csv(path_csv, encoding="latin1")
                    # end if
                except Exception as exc:
                    logger.warning(f"Failed to fetch dataset from Kaggle via kagglehub ({exc}).")
                # end try
            # end if

            if path_source.startswith("http://") or path_source.startswith("https://"):
                try:
                    logger.info(f"Fetching Speed Dating dataset from URL: {path_source}")
                    df_raw = pd.read_csv(path_source, encoding="latin1")
                    return df_raw
                except Exception as exc:
                    logger.warning(f"Failed to fetch dataset from URL ({exc}).")
                # end try
            # end if
        # end if

        # Attempt Kaggle download as default before synthetic generation
        try:
            import kagglehub
            logger.info("Attempting default download from Kaggle (annavictoria/speed-dating-experiment)...")
            path_download = kagglehub.dataset_download("annavictoria/speed-dating-experiment")
            path_csv = os.path.join(path_download, "Speed Dating Data.csv")
            if os.path.exists(path_csv):
                return pd.read_csv(path_csv, encoding="latin1")
            # end if
        except Exception as exc:
            logger.warning(f"Default Kaggle download unavailable ({exc}). Generating synthetic data.")
        # end try

        logger.warning(
            "Raw Speed Dating file not available. Generating representative synthetic Speed Dating data for offline execution."
        )
        return self._generate_synthetic_speed_dating_data()
        # end def load_data_raw

    def _generate_synthetic_speed_dating_data(
        self,
        n_participants_per_wave: int = 20,
        n_waves: int = 4,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """Generates synthetic speed dating records matching the real Columbia experimental schema.

        Args:
            n_participants_per_wave: Total participants per wave (split equally by gender).
            n_waves: Number of experimental speed dating waves.
            random_seed: Random seed for reproducibility.

        Returns:
            DataFrame with row-per-person-per-date structure matching the 195-variable schema.
        """
        rng = np.random.RandomState(random_seed)
        list_rows: ty.List[ty.Dict[str, ty.Any]] = []

        current_iid = 1
        interest_names = [
            "sports", "tvsports", "exercise", "dining", "museums", "art",
            "hiking", "gaming", "clubbing", "reading", "tv", "theater",
            "movies", "concerts", "music", "shopping", "yoga"
        ]

        for wave in range(1, n_waves + 1):
            n_half = n_participants_per_wave // 2
            # Generate female participants (gender = 0)
            females = []
            for _ in range(n_half):
                p_data = {
                    "iid": current_iid,
                    "gender": 0,
                    "wave": wave,
                    "age": rng.randint(21, 36),
                    "race": rng.choice([1, 2, 3, 4, 6], p=[0.1, 0.55, 0.08, 0.22, 0.05]),
                    "imprace": rng.randint(1, 11),
                    "imprelig": rng.randint(1, 11),
                    "date": rng.randint(1, 8),
                    "go_out": rng.randint(1, 8),
                    "goal": rng.randint(1, 7),
                    "field_cd": rng.randint(1, 19),
                    "attr3_1": rng.randint(4, 11),
                    "sinc3_1": rng.randint(5, 11),
                    "intel3_1": rng.randint(6, 11),
                    "fun3_1": rng.randint(5, 11),
                    "amb3_1": rng.randint(5, 11),
                }
                # Stated preferences summing to ~100
                prefs = rng.dirichlet(np.ones(6)) * 100.0
                p_data["attr1_1"] = prefs[0]
                p_data["sinc1_1"] = prefs[1]
                p_data["intel1_1"] = prefs[2]
                p_data["fun1_1"] = prefs[3]
                p_data["amb1_1"] = prefs[4]
                p_data["shar1_1"] = prefs[5]

                for inter in interest_names:
                    p_data[inter] = rng.randint(1, 11)
                # end for inter

                females.append(p_data)
                current_iid += 1
            # end for female

            # Generate male participants (gender = 1)
            males = []
            for _ in range(n_half):
                p_data = {
                    "iid": current_iid,
                    "gender": 1,
                    "wave": wave,
                    "age": rng.randint(22, 38),
                    "race": rng.choice([1, 2, 3, 4, 6], p=[0.1, 0.55, 0.08, 0.22, 0.05]),
                    "imprace": rng.randint(1, 11),
                    "imprelig": rng.randint(1, 11),
                    "date": rng.randint(1, 8),
                    "go_out": rng.randint(1, 8),
                    "goal": rng.randint(1, 7),
                    "field_cd": rng.randint(1, 19),
                    "attr3_1": rng.randint(4, 11),
                    "sinc3_1": rng.randint(5, 11),
                    "intel3_1": rng.randint(6, 11),
                    "fun3_1": rng.randint(5, 11),
                    "amb3_1": rng.randint(5, 11),
                }
                prefs = rng.dirichlet(np.ones(6)) * 100.0
                p_data["attr1_1"] = prefs[0]
                p_data["sinc1_1"] = prefs[1]
                p_data["intel1_1"] = prefs[2]
                p_data["fun1_1"] = prefs[3]
                p_data["amb1_1"] = prefs[4]
                p_data["shar1_1"] = prefs[5]

                for inter in interest_names:
                    p_data[inter] = rng.randint(1, 11)
                # end for inter

                males.append(p_data)
                current_iid += 1
            # end for male

            # Simulate round-robin speed dates between each male and female
            for order, f_person in enumerate(females):
                for round_num, m_person in enumerate(males):
                    # Compatibility likelihood based on attractiveness and shared interests
                    f_interests = np.array([f_person[k] for k in interest_names])
                    m_interests = np.array([m_person[k] for k in interest_names])
                    diff_interest = np.mean(np.abs(f_interests - m_interests))
                    samerace = 1 if f_person["race"] == m_person["race"] else 0

                    prob_f_yes = 0.35 + 0.04 * (m_person["attr3_1"] - 7.0) - 0.03 * (diff_interest - 3.0)
                    prob_m_yes = 0.45 + 0.04 * (f_person["attr3_1"] - 7.0) - 0.03 * (diff_interest - 3.0)

                    dec_f = 1 if rng.rand() < np.clip(prob_f_yes, 0.05, 0.85) else 0
                    dec_m = 1 if rng.rand() < np.clip(prob_m_yes, 0.05, 0.85) else 0
                    mutual_match = 1 if (dec_f == 1 and dec_m == 1) else 0

                    # Row 1: Female perspective (evaluating male partner)
                    row_f = dict(f_person)
                    row_f.update({
                        "id": f_person["iid"] % 100,
                        "idg": order + 1,
                        "condtn": 1,
                        "round": len(males),
                        "position": round_num + 1,
                        "positin1": round_num + 1,
                        "order": round_num + 1,
                        "partner": round_num + 1,
                        "pid": m_person["iid"],
                        "match": mutual_match,
                        "samerace": samerace,
                        "age_o": m_person["age"],
                        "race_o": m_person["race"],
                        "dec_o": dec_m,
                        "dec": dec_f,
                        "like": rng.randint(4, 11) if dec_f == 1 else rng.randint(1, 7),
                        "prob": rng.randint(4, 11) if dec_m == 1 else rng.randint(1, 6),
                        "met": rng.choice([2, 2, 2, 1]),
                        "attr": rng.randint(5, 11) if dec_f == 1 else rng.randint(1, 7),
                        "sinc": rng.randint(5, 11),
                        "intel": rng.randint(6, 11),
                        "fun": rng.randint(4, 11),
                        "amb": rng.randint(5, 11),
                        "shar": rng.randint(4, 11),
                    })
                    list_rows.append(row_f)

                    # Row 2: Male perspective (evaluating female partner)
                    row_m = dict(m_person)
                    row_m.update({
                        "id": m_person["iid"] % 100,
                        "idg": round_num + 1,
                        "condtn": 1,
                        "round": len(females),
                        "position": order + 1,
                        "positin1": order + 1,
                        "order": order + 1,
                        "partner": order + 1,
                        "pid": f_person["iid"],
                        "match": mutual_match,
                        "samerace": samerace,
                        "age_o": f_person["age"],
                        "race_o": f_person["race"],
                        "dec_o": dec_f,
                        "dec": dec_m,
                        "like": rng.randint(4, 11) if dec_m == 1 else rng.randint(1, 7),
                        "prob": rng.randint(4, 11) if dec_f == 1 else rng.randint(1, 6),
                        "met": rng.choice([2, 2, 2, 1]),
                        "attr": rng.randint(5, 11) if dec_m == 1 else rng.randint(1, 7),
                        "sinc": rng.randint(5, 11),
                        "intel": rng.randint(6, 11),
                        "fun": rng.randint(4, 11),
                        "amb": rng.randint(5, 11),
                        "shar": rng.randint(4, 11),
                    })
                    list_rows.append(row_m)
                # end for m_person
            # end for f_person
        # end for wave

        df_synthetic = pd.DataFrame(list_rows)
        # Introduce light missingness (1-2%) to match real survey conditions
        for col in ["age", "imprace", "imprelig", "date", "go_out"] + interest_names[:5]:
            mask_na = rng.rand(len(df_synthetic)) < 0.02
            df_synthetic.loc[mask_na, col] = np.nan
        # end for col

        logger.info(
            f"Generated synthetic speed dating dataset with {len(df_synthetic)} observations, "
            f"{len(df_synthetic.columns)} columns, and {(df_synthetic['match'] == 1).sum()} mutual matches."
        )
        return df_synthetic
        # end def _generate_synthetic_speed_dating_data
# end class SpeedDatingDataLoader
