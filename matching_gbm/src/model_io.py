import joblib

# Everything inference needs beyond the fitted estimator travels with it in one
# file. The category map in particular must be the *same* mapping used at fit
# time — rebuilding it from whatever items happen to be in the test set would
# silently renumber the categories and feed the model garbage codes.
BUNDLE_KEYS = ("model", "category_map", "feature_names")


def save_bundle(path, model, category_map, feature_names):
    joblib.dump(
        {"model": model, "category_map": category_map, "feature_names": list(feature_names)},
        path,
    )


def load_bundle(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj.get("category_map"), obj.get("feature_names")
    # A bare estimator: a model saved before the bundle format existed.
    return obj, None, None
