"""Cache one site: read every labelled image straight from its zip, crop and resize it, save it as JPEG.

    python scripts/prepare_data.py --site odir

Writes data/cache/<site>/<image_id>.jpg and data/meta/<site>.csv. Images already in the cache are skipped,
so an interrupted run can be restarted. The archive itself is not modified (download_data.sh deletes it
after a complete cache).
"""
import argparse
import io
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from retina_shift.preprocess import prepare_image
from retina_shift.sites import site_table
from retina_shift.splits import META_DIR, ROOT

CACHE_DIR = ROOT / "data" / "cache"
_archive = None


def _open(path: str) -> None:
    global _archive
    _archive = zipfile.ZipFile(path)  # one handle per worker process


def _convert(job: tuple[str, str]) -> None:
    member, out_path = job
    img = Image.open(io.BytesIO(_archive.read(member)))
    prepare_image(img).save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--zip-dir", default=str(ROOT / "data" / "zips"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    table = site_table(args.site, Path(args.zip_dir))
    out_dir = CACHE_DIR / args.site
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(m, str(out_dir / f"{i}.jpg")) for m, i in zip(table["member"], table["image_id"])
            if not (out_dir / f"{i}.jpg").exists()]
    print(f"{args.site}: {len(table)} labelled images, {len(jobs)} to convert")
    with Pool(args.workers, initializer=_open, initargs=(str(Path(args.zip_dir) / table["zip"].iloc[0]),)) as pool:
        for k, _ in enumerate(pool.imap_unordered(_convert, jobs, chunksize=16), 1):
            if k % 5000 == 0:
                print(f"  {k}/{len(jobs)}")
    missing = [i for i in table["image_id"] if not (out_dir / f"{i}.jpg").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing from the cache, e.g. {missing[:3]}")
    META_DIR.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["member", "zip"]).to_csv(META_DIR / f"{args.site}.csv", index=False)
    print(f"{args.site}: cache complete, metadata written")


if __name__ == "__main__":
    main()
