import numpy as np
from data_analysis_variable_selection.prototype.extractor import PrototypeExtractor


def test_prototype_extractor():
    np.random.seed(42)
    n = 40
    d = 4
    x = np.random.randn(n, d)
    y = np.random.randn(n, d) + 2.0  # Shifted

    names = ["f0", "f1", "f2", "f3"]
    extractor = PrototypeExtractor(metric_mode="discriminative")

    res = extractor.extract_samples_prototype(
        sample_x=x,
        sample_y=y,
        names_features=names,
        list_subspace_indices=[0, 1],
        type_subspace="hat_S",
        top_n=3
    )

    assert len(res.list_prototype_records) == 6  # 3 for X, 3 for Y
    prototypes_x = [p for p in res.list_prototype_records if p.label_class == "X"]
    prototypes_y = [p for p in res.list_prototype_records if p.label_class == "Y"]

    assert len(prototypes_x) == 3
    assert len(prototypes_y) == 3
    assert prototypes_x[0].type_subspace == "hat_S"
    assert "f0" in prototypes_x[0].dict_feature_values
    assert "f1" in prototypes_x[0].dict_feature_values
