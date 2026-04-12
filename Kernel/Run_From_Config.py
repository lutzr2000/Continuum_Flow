import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) >= 2:
        config_path = Path(sys.argv[1]).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = json.load(sys.stdin)

    kernel_dir = Path(__file__).resolve().parent
    if str(kernel_dir) not in sys.path:
        sys.path.insert(0, str(kernel_dir))

    import Kernel_GPU

    Kernel_GPU.main(config)


if __name__ == "__main__":
    main()
