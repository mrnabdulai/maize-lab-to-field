"""Train on lab maize photos, test on field photos, with and without diffusion-translated images.

Classifier: frozen ImageNet ResNet-50 features + logistic regression (light enough for an 8 GB laptop).
Conditions (all tested on every PlantDoc field image):
  lab            450 PlantVillage lab photos
  lab+augment    plus one standard augmentation of each (crop, flip, rotate, colour jitter)
  lab+diffusion v1  plus one lab-to-field translation of each (translate.py v1: wide scene, disease in prompt)
  lab+diffusion v2  plus one lab-to-field translation of each (translate.py v2: close-up, no disease in prompt)
Reference: 5-fold cross-validation on the field photos themselves (what real field data gets).
Metric: balanced accuracy (mean per-class recall), since field classes are uneven; 95% bootstrap CI.

Run: python3 evaluate.py   -> results.json, chart.png
"""
import json, os
import numpy as np
import torch, torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from torchvision.transforms import v2

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["gray_leaf_spot", "leaf_blight", "rust"]
torch.set_num_threads(4)
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
net = torchvision.models.resnet50(weights=weights).eval()
net.fc = torch.nn.Identity()
prep = weights.transforms()
augment = v2.Compose([v2.RandomResizedCrop(256, scale=(0.5, 1.0)), v2.RandomHorizontalFlip(),
                      v2.RandomRotation(20), v2.ColorJitter(0.4, 0.4, 0.4, 0.05)])


def load(split):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        d = os.path.join(HERE, "data", split, c)
        names = sorted(os.listdir(d))
        paths += [os.path.join(d, n) for n in names]
        labels += [i] * len(names)
    return paths, np.array(labels)


@torch.no_grad()
def features(paths, aug=False, cache=None):
    path = os.path.join(HERE, "data", f"features_{cache}.npy") if cache else None
    if path and os.path.exists(path):  # ponytail: cache by name; delete data/features_*.npy if images change
        return np.load(path)
    torch.manual_seed(0)
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 16]]
        out.append(net(torch.stack([prep(augment(im) if aug else im) for im in ims])).numpy())
    out = np.concatenate(out)
    if path:
        np.save(path, out)
    return out


def fit(X, y):
    return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(X, y)


def ci(y, pred, n=2000):
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    s = [balanced_accuracy_score(y[i], pred[i]) for i in idx]
    return [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]


def main():
    lab_p, lab_y = load("lab")
    v1_p, v1_y = load("translated_v1")
    v2_p, v2_y = load("translated_v2")
    field_p, field_y = load("field")
    print("lab", len(lab_p), "v1", len(v1_p), "v2", len(v2_p), "field", len(field_p), flush=True)
    X_lab, X_aug = features(lab_p, cache="lab"), features(lab_p, aug=True, cache="lab_aug")
    X_v1, X_v2, X_field = features(v1_p, cache="v1"), features(v2_p, cache="v2"), features(field_p, cache="field")

    results = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lab_cv = cross_val_predict(fit(X_lab, lab_y), X_lab, lab_y, cv=cv)
    results["lab photos, tested on held-out lab photos (5-fold)"] = {"balanced_accuracy": round(balanced_accuracy_score(lab_y, lab_cv), 3)}
    for name, X, y in [("lab", X_lab, lab_y),
                       ("lab+augment", np.vstack([X_lab, X_aug]), np.concatenate([lab_y, lab_y])),
                       ("lab+diffusion v1", np.vstack([X_lab, X_v1]), np.concatenate([lab_y, v1_y])),
                       ("lab+diffusion v2", np.vstack([X_lab, X_v2]), np.concatenate([lab_y, v2_y]))]:
        pred = fit(X, y).predict(X_field)
        results[name] = {"balanced_accuracy": round(balanced_accuracy_score(field_y, pred), 3), "ci95": ci(field_y, pred),
                         "accuracy": round(float((pred == field_y).mean()), 3),
                         "confusion (rows true, cols predicted)": confusion_matrix(field_y, pred).tolist()}
    field_cv = cross_val_predict(fit(X_field, field_y), X_field, field_y, cv=cv)
    results["field photos, 5-fold (reference)"] = {"balanced_accuracy": round(balanced_accuracy_score(field_y, field_cv), 3),
                                                   "ci95": ci(field_y, field_cv)}
    results["classes"] = CLASSES
    results["field images per class"] = np.bincount(field_y).tolist()
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
