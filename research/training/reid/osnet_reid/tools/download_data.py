"""
Download SoccerNet ReID dataset.

Usage:
    python3 tools/download_data.py --dest datasets --splits train valid test challenge

SoccerNet requires a password obtained by signing the NDA at:
    https://www.soccer-net.org/data
"""
import argparse
import os
import sys

VALID_SPLITS = ['train', 'valid', 'test', 'challenge']


def download(dest: str, splits: list, password: str):
    from SoccerNet.Downloader import SoccerNetDownloader

    os.makedirs(dest, exist_ok=True)
    dl = SoccerNetDownloader(LocalDirectory=dest)
    dl.password = password

    print(f"Downloading splits: {splits} → {os.path.abspath(dest)}")
    dl.downloadDataTask(task='reid', split=splits)
    print("Download complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', default='datasets',
                        help='Root directory for datasets (default: datasets)')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test', 'challenge'],
                        choices=VALID_SPLITS,
                        help='Dataset splits to download')
    parser.add_argument('--password', default='',
                        help='SoccerNet password (or set env var SOCCERNET_PASSWORD)')
    args = parser.parse_args()

    password = args.password or os.environ.get('SOCCERNET_PASSWORD', '')
    if not password:
        print("ERROR: SoccerNet password required.")
        print("  Get it by signing the NDA at https://www.soccer-net.org/data")
        print("  Then pass it via --password or SOCCERNET_PASSWORD env var.")
        sys.exit(1)

    download(dest=args.dest, splits=args.splits, password=password)


if __name__ == '__main__':
    main()
