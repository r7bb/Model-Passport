"""Demo evaluate stage. The logic is generic: see ``model_passport.ml.stages.evaluate``."""

from model_passport.ml.stages import run

if __name__ == "__main__":
    run("evaluate", {"label": "income"})
