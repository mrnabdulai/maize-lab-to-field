"""Download maize leaf images: lab photos (PlantVillage) and field photos (PlantDoc).

Lab:   https://github.com/spMohanty/PlantVillage-Dataset (raw/color), 150 random images per class.
Field: https://github.com/pratikkayal/PlantDoc-Dataset (train + test), every maize image.
Writes data/lab/<class>/ and data/field/<class>/. Needs the gh CLI.
"""
import json, os, random, subprocess, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

CLASSES = {  # our label: (PlantVillage folder, PlantDoc folder)
    "gray_leaf_spot": ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot"),
    "leaf_blight": ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight"),
    "rust": ("Corn_(maize)___Common_rust_", "Corn rust leaf"),
}
LAB_PER_CLASS = 150
HERE = os.path.dirname(os.path.abspath(__file__))


def ls(repo, path):
    out = subprocess.run(["gh", "api", f"repos/{repo}/contents/{urllib.parse.quote(path)}"],
                         capture_output=True, text=True, check=True).stdout
    return [f["name"] for f in json.loads(out) if f["type"] == "file"]


def fetch(job):
    url, dest = job
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)


def main():
    jobs = []
    for label, (pv, pd) in CLASSES.items():
        for split, dirs in [("lab", [("spMohanty/PlantVillage-Dataset", "master", f"raw/color/{pv}")]),
                            ("field", [("pratikkayal/PlantDoc-Dataset", "master", f"{s}/{pd}") for s in ("train", "test")])]:
            out = os.path.join(HERE, "data", split, label)
            os.makedirs(out, exist_ok=True)
            for repo, branch, path in dirs:
                names = sorted(ls(repo, path))
                if split == "lab":
                    names = random.Random(0).sample(names, LAB_PER_CLASS)
                for n in names:
                    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{urllib.parse.quote(path)}/{urllib.parse.quote(n)}"
                    jobs.append((url, os.path.join(out, f"{path.split('/')[0]}_{n}".replace(" ", "_"))))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, jobs))
    for split in ("lab", "field"):
        for label in CLASSES:
            print(split, label, len(os.listdir(os.path.join(HERE, "data", split, label))))


if __name__ == "__main__":
    main()
