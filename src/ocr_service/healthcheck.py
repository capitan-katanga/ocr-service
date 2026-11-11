"""HEALTHCHECK de la imagen y del compose: sale 0 solo si /health/ready da 200. Solo stdlib."""

import sys
import urllib.request


def main(url: str = "http://127.0.0.1:8084/health/ready") -> int:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:  # noqa: S310 - URL fija de loopback
            return 0 if r.status == 200 else 1
    except OSError:
        return 1


if __name__ == "__main__":
    sys.exit(main())
