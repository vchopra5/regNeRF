"""Package only the saved files required by the toy lesson. Do not train."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from notebooks.toy_experiment import ASSETS, saved_asset_names


def main():
    payloads = {name: (ASSETS / name).read_bytes() for name in sorted(saved_asset_names())}
    manifest = {"version": 1, "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    payloads["manifest.json"] = json.dumps(manifest, indent=2).encode()
    bundle = Path(__file__).with_name("toy_saved_results.zip")
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in payloads.items():
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    print(f"Packaged {len(manifest['files'])} saved files in {bundle.name}: {bundle.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
