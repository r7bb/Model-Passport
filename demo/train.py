"""Demo train stage. The logic is generic: see ``model_passport.ml.stages.train``."""

from model_passport.ml.stages import run

if __name__ == "__main__":
    run("train", {"label": "income"})
