import typing as ty
from pydantic import BaseModel, Field, ConfigDict


class SpeedDatingPreprocessingConfig(BaseModel):
    """Configuration parameters for Speed Dating dataset ingestion, preprocessing, and reporting.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path_data_file: ty.Optional[str] = Field(
        default=None,
        description="Local filesystem path to raw Speed Dating CSV file."
    )
    url_download: ty.Optional[str] = Field(
        default="https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment",
        description="Source URL for downloading the raw Speed Dating dataset from Kaggle."
    )
    max_records_per_distribution: ty.Optional[int] = Field(
        default=500,
        description="Maximum number of reciprocal date pairs retained per distribution."
    )
    random_seed_sampling: int = Field(
        default=42,
        description="Random seed for reproducible subsampling."
    )
    include_interaction_deltas: bool = Field(
        default=True,
        description="Whether to generate homophily delta and cosine similarity features."
    )
    include_post_date_evaluations: bool = Field(
        default=False,
        description="Whether to include post-date ratings (e.g. like, attr). Default False to avoid evaluative tautology."
    )
    columns_demographics: ty.List[str] = Field(
        default_factory=lambda: ["age", "imprace", "imprelig", "date", "go_out", "goal", "exphappy", "expnum", "career_c"],
        description="Demographic, lifestyle, career, and expectation survey features."
    )
    columns_interests: ty.List[str] = Field(
        default_factory=lambda: [
            "sports", "tvsports", "exercise", "dining", "museums", "art",
            "hiking", "gaming", "clubbing", "reading", "tv", "theater",
            "movies", "concerts", "music", "shopping", "yoga"
        ],
        description="17 leisure and lifestyle interest ratings (1-10 scale)."
    )
    columns_self_ratings: ty.List[str] = Field(
        default_factory=lambda: ["attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1"],
        description="Self-perception ratings (1-10 scale)."
    )
    columns_stated_preferences: ty.List[str] = Field(
        default_factory=lambda: ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"],
        description="Stated mate preference weights."
    )
    columns_administrative_to_drop: ty.List[str] = Field(
        default_factory=lambda: [
            "id", "idg", "condtn", "wave", "round", "position", "positin1", "order", "partner"
        ],
        description="Experimental tracking columns to exclude from analytical feature space."
    )
    columns_post_date_ratings: ty.List[str] = Field(
        default_factory=lambda: [
            "attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob", "met"
        ],
        description="Post-date evaluative ratings."
    )
# end class SpeedDatingPreprocessingConfig
