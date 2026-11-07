import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_ROOT, "data")
CHAMPIONS_DIR = os.path.join(DATA_DIR, "champions")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

for _d in (DATA_DIR, CHAMPIONS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_CHAMPION = os.path.join(CHAMPIONS_DIR, "champion.json")
COMPARISON_RESULTS = os.path.join(DATA_DIR, "comparison_results.json")


def champion(name):
    if os.path.dirname(name):  # already a path -- leave it alone
        return name
    if not name.startswith("champion"):
        name = f"champion_{name}"
    if not name.endswith(".json"):
        name += ".json"
    return os.path.join(CHAMPIONS_DIR, name)


def log(name):
    if os.path.dirname(name):
        return name
    if not name.startswith("training_log"):
        name = f"training_log_{name}"
    if not name.endswith(".csv"):
        name += ".csv"
    return os.path.join(LOGS_DIR, name)
