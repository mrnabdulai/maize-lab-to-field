"""Leaf masks for PlantVillage lab photos: the leaf is coloured, the backdrop is dark or grey."""
import numpy as np
from PIL import Image
from scipy import ndimage


def leaf_mask(path):
    """True on the leaf. Backdrop = dark or grey pixels connected to the image border, so pale
    lesions inside the leaf are never treated as backdrop. The leaf is shrunk by 3 px so no backdrop
    survives at its edge: rust photos have black backdrops and the others grey, a cue the model could learn."""
    hsv = np.asarray(Image.open(path).convert("HSV")).astype(int)
    bg = ndimage.binary_opening((hsv[..., 2] < 40) | (hsv[..., 1] < 28), iterations=2)
    lab, _ = ndimage.label(bg)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge)) | (hsv[..., 2] < 30)  # black backdrop showing through holes in the leaf
    return ndimage.binary_erosion(~bg, iterations=3)


if __name__ == "__main__":  # contact sheet: 4 images per class, image over mask
    import os, sys
    sheet = Image.new("RGB", (256 * 12, 512))
    for c, label in enumerate(["gray_leaf_spot", "leaf_blight", "rust"]):
        d = os.path.join("data", "lab", label)
        for i, n in enumerate(sorted(os.listdir(d))[:4]):
            p, x = os.path.join(d, n), 256 * (4 * c + i)
            sheet.paste(Image.open(p).convert("RGB").resize((256, 256)), (x, 0))
            sheet.paste(Image.fromarray((leaf_mask(p) * 255).astype("uint8")).resize((256, 256)).convert("RGB"), (x, 256))
    sheet.resize((1536, 256)).save(sys.argv[1] if len(sys.argv) > 1 else "mask_check.png")
