"""Demo preprocess stage. The logic is generic: see ``model_passport.ml.stages.preprocess``."""

from model_passport.ml.stages import run

if __name__ == "__main__":
    run("preprocess", {"label": "income"})
