"""The wide universe's command line.

    python -m scaata.universe refresh    download ~200 stocks and freeze the snapshot (monthly is plenty)
    python -m scaata.universe study      walk-forward test of a model that learns from scratch
    python -m scaata.universe book       advance the wide paper book by the sessions it has not seen

The wide paper book is separate from the five-stock agent: its own file, its own evidence meter, no shared state.
"""
from __future__ import annotations

import sys


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    command = argv[0] if argv else ""
    if command == "refresh":
        from scaata.universe import data

        closes = data.download()
        info = data.freeze(closes)
        print(f"{info['stocks']} stocks, {info['first']} to {info['last']}, {info['with_history']} with enough history; "
              f"missing: {', '.join(info['missing']) or 'none'}")
    elif command == "study":
        from scaata.universe import study

        out = study.run()
        print(f"{out['stocks']} stocks, {out['weeks_scored']} weeks scored")
        for span in ("all", "2010-2019", "2020 onward"):
            row = out["ic"][span]
            print(f"  {span:<12} IC  " + "  ".join(f"{k} {v['mean']:+.4f}" for k, v in row.items() if v["mean"] is not None))
            for name, v in out["versus"][span].items():
                if v["mean"] is not None:
                    print(f"      {name:<22} {v['mean']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}] p={v['p']}")
    elif command == "book":
        from scaata.universe import book

        print(book.advance())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
