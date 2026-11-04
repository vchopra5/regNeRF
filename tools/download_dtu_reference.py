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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='Points.zip')
    parser.add_argument('--match', default='stl040_total.ply')
    args = parser.parse_args()
    url = 'https://roboimagedata2.compute.dtu.dk/data/MVS/' + args.archive
    with zipfile.ZipFile(RemoteZip(url)) as archive:
        matches = [n for n in archive.namelist() if args.match.lower() in n.lower()]
        print(matches, flush=True)
        for name in matches:
            info = archive.getinfo(name)
            print(name, info.file_size, 'bytes', flush=True)
            output = Path('data/dtu_reference') / (Path(name).name if name.endswith('.ply') else name.replace('/', '__'))
            if not output.exists():
                output.write_bytes(archive.read(name))
            print(output, flush=True)
