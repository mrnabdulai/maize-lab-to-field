"""Lab-to-field translation: Stable Diffusion 1.5 inpainting on Cloudflare Workers AI.

The leaf pixels are kept exactly (so the lesions stay real) and everything else is painted as a
maize field. Runs on Cloudflare, not locally. Needs `npx wrangler login` and CLOUDFLARE_ACCOUNT_ID.

v1: leaf shrunk to 320 px on a 512 px canvas (wide scene); prompt names the disease.
v2: leaf kept at full size, so only the lab backdrop is repainted (close-up, like most field
    photos); prompt does not name the disease, so the model has no reason to paint lesions.

Run: python3 translate.py v1|v2                       -> data/translated_<version>/<class>/
     python3 translate.py v1|v2 preview <image> ...   -> preview_<class>.png (lab photo next to its translation)
"""
import io, json, os, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from masks import leaf_mask

ACCOUNT = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
MODEL = "@cf/runwayml/stable-diffusion-v1-5-inpainting"
HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["gray_leaf_spot", "leaf_blight", "rust"]
DISEASE = {"gray_leaf_spot": "grey leaf spot lesions", "leaf_blight": "northern leaf blight lesions", "rust": "common rust pustules"}
VERSIONS = {
    "v1": {"leaf_size": 320, "prompt": "smartphone photo of a maize leaf with {} on a living maize plant in a green maize field, "
                                       "natural sunlight, rows of maize plants, leaves and soil in the background"},
    "v2": {"leaf_size": 512, "prompt": "close-up smartphone photo of a maize leaf on a living maize plant in a green maize field, "
                                       "natural sunlight, out-of-focus maize plants and soil in the background"},
}
NEGATIVE = "plain background, studio, black background, grey background, road, concrete, wall, drawing, cartoon"
_token = None


def token(refresh=False):
    global _token
    if refresh or _token is None:
        _token = subprocess.run(["npx", "-y", "wrangler@latest", "auth", "token"],
                                capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return _token


def translate(path, label, version, seed=0):
    size, prompt = VERSIONS[version]["leaf_size"], VERSIONS[version]["prompt"].format(DISEASE[label])
    off = (512 - size) // 2
    img = Image.new("RGB", (512, 512), (110, 100, 80))
    img.paste(Image.open(path).convert("RGB").resize((size, size)), (off, off))
    keep = Image.fromarray((leaf_mask(path) * 255).astype("uint8")).resize((size, size))
    mask = Image.new("L", (512, 512), 255)  # white = repaint
    mask.paste(Image.eval(keep, lambda v: 255 - v), (off, off))
    png = lambda im: (lambda b: (im.save(b, "PNG"), list(b.getvalue()))[1])(io.BytesIO())
    body = json.dumps({"prompt": prompt, "negative_prompt": NEGATIVE, "image": png(img), "mask": png(mask),
                       "strength": 1.0, "num_steps": 20, "guidance": 7.5, "seed": seed}).encode()
    for attempt in range(5):
        req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{MODEL}", body,
                                     {"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
        try:
            out = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=120).read())).convert("RGB").resize((512, 512))
            return Image.composite(out, img, mask)  # paste the real leaf back: SD inpainting also alters kept pixels
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            print(f"  {e.code} {msg}", file=sys.stderr, flush=True)
            if e.code in (401, 403):
                token(refresh=True)
            elif e.code == 429 and b"daily" in msg.lower():
                raise SystemExit("daily Workers AI limit reached; rerun tomorrow, finished images are kept")
        except Exception as e:  # ponytail: plain retry with backoff for network blips
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr, flush=True)
        time.sleep(10 * (attempt + 1))
    return None


def main(version):
    jobs = []
    for label in CLASSES:
        src, out = os.path.join(HERE, "data", "lab", label), os.path.join(HERE, "data", f"translated_{version}", label)
        os.makedirs(out, exist_ok=True)
        jobs += [(os.path.join(src, n), label, os.path.join(out, n.rsplit(".", 1)[0] + ".jpg"))
                 for n in sorted(os.listdir(src)) if not os.path.exists(os.path.join(out, n.rsplit(".", 1)[0] + ".jpg"))]
    print(f"{len(jobs)} images to translate", flush=True)

    def run(job):
        src, label, dest = job
        img = translate(src, label, version)
        if img:
            img.resize((256, 256)).save(dest, quality=92)
        print("ok" if img else "FAILED", dest, flush=True)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run, jobs))


if __name__ == "__main__":
    version = sys.argv[1]
    if sys.argv[2:3] == ["preview"]:
        for p in sys.argv[3:]:
            label = os.path.basename(os.path.dirname(p))
            row = [Image.open(p).convert("RGB").resize((256, 256)), translate(p, label, version).resize((256, 256))]
            sheet = Image.new("RGB", (512, 256))
            for i, im in enumerate(row):
                sheet.paste(im, (256 * i, 0))
            sheet.save(os.path.join(HERE, f"preview_{label}.png"))
            print("saved preview", label)
    else:
        main(version)
