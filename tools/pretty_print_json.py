import json
import sys
from pathlib import Path

def pretty_print(path: str):
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    with p.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python pretty_print_json.py <file.json>')
        sys.exit(2)
    pretty_print(sys.argv[1])
