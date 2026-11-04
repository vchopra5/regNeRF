"""Read one scan from the official DTU archive with HTTP range requests."""
import io
from pathlib import Path
import urllib.request
import zipfile


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=-65536'}), timeout=30) as r:
            if r.status != 206:
                raise RuntimeError('The server must support range requests.')
            self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size <= 0:
            return b''
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{self.position+size-1}'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError('Unexpected full archive response.')
            data = response.read()
        self.position += len(data)
        return data


def safe_name(member):
    """Flatten a zip member name to one file name. Reject names that could leave the output folder."""
    name = member.replace('\\', '/').strip('/')
    parts = name.split('/')
    if not name or any(part in ('', '.', '..') for part in parts) or ':' in name:
        raise ValueError(f'Unsafe archive member name: {member!r}')
    return parts[-1] if name.lower().endswith('.ply') else '__'.join(parts)


if __name__ == '__main__':
    import argparse
    import hashlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='Points.zip')
    parser.add_argument('--match', default='stl040_total.ply')
    parser.add_argument('--sha256', help='Expected SHA-256 of the extracted file. Stop if it differs.')
    args = parser.parse_args()
    # HTTPS with the default certificate checks of urllib.
    url = 'https://roboimagedata2.compute.dtu.dk/data/MVS/' + args.archive
    folder = Path('data/dtu_reference')
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZip(url)) as archive:
        matches = [n for n in archive.namelist() if args.match.lower() in n.lower()]
        print(matches, flush=True)
        if args.sha256 and len(matches) != 1:
            raise SystemExit('--sha256 needs exactly one matching file.')
        for name in matches:
            info = archive.getinfo(name)
            print(name, info.file_size, 'bytes', flush=True)
            output = folder / safe_name(name)
            if output.exists():
                data = output.read_bytes()
            else:
                data = archive.read(name)  # zipfile checks the CRC-32 of each member.
            digest = hashlib.sha256(data).hexdigest()
            if args.sha256 and digest != args.sha256.lower():
                raise SystemExit(f'SHA-256 mismatch for {name}: {digest}. The file was not saved.')
            if not output.exists():
                temporary = output.with_suffix(output.suffix + '.part')
                temporary.write_bytes(data)
                temporary.replace(output)
            print(output, 'sha256', digest, flush=True)
