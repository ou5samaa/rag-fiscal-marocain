"""Installer la distribution portable officielle, sans modifier le PATH système."""
from datetime import datetime,timezone
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rag_fiscal.common import ROOT,sha256,write_json,configure_console

VERSION='0.40.1'
DIGEST='b394d14436d38032f23190e3f14eb2c6dad5ebbe4e192414f74c8fdca01703ab'
URL=f'https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-windows-amd64.zip'


def main():
    configure_console()
    dest=ROOT/'runtime/ollama'/VERSION
    if dest.exists():raise SystemExit('Dossier existant : ne pas remplacer une installation en place.')
    archive=ROOT/'runtime'/f'ollama-{VERSION}.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    print('Téléchargement Ollama '+VERSION,flush=True)
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=60) as response,archive.open('wb') as output:
            last=0;total=0
            while block:=response.read(4*1024*1024):
                output.write(block);total+=len(block)
                if total-last>=256*1024*1024:print(f'{total/1e9:.2f} Go téléchargés',flush=True);last=total
    if sha256(archive)!=DIGEST:raise ValueError('Archive incomplète ou empreinte officielle différente.')
    dest.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if not (dest/entry.filename).resolve().is_relative_to(dest.resolve()):raise ValueError('Chemin archive invalide')
        z.extractall(dest)
    files={p.relative_to(dest).as_posix():sha256(p) for p in dest.rglob('*') if p.is_file()}
    write_json(dest/'installation.json',{'version':VERSION,'source_url':URL,'archive_sha256':DIGEST,
        'installed_at':datetime.now(timezone.utc).isoformat(),'files_sha256':files})
    assert archive.resolve().is_relative_to((ROOT/'runtime').resolve())
    archive.unlink()  # Archive de téléchargement vérifiée, devenue redondante.
    print(dest,flush=True)


if __name__=='__main__':main()
