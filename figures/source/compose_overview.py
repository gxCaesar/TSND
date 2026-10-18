#!/usr/bin/env python3
"""Compile the original native TSND artwork and its cropped vector composition.

Run from the manuscript directory:
    python3 figures/source/compose_overview.py

Requires Python 3 and a TeX distribution with pdflatex, standalone, newtxtext,
graphicx, and TikZ. The editable .tikz documents are copied to temporary .tex
files only for compilation. No image-generation or raster conversion step is used.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> None:
    source = Path(__file__).resolve().parent
    names = ("fig1_motivation", "fig2_pipeline", "fig1_original_composite")
    with tempfile.TemporaryDirectory(prefix="tsnd-native-fig1-") as tmp:
        build = Path(tmp)
        for name in names:
            shutil.copy2(source / f"{name}.tikz", build / f"{name}.tex")
        for name in names:
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-no-shell-escape",
                    f"{name}.tex",
                ],
                cwd=build,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"{name} failed to compile:\n{result.stdout}")
            print(f"Compiled {name}.tikz")
        for name in names[:2]:
            shutil.copy2(build / f"{name}.pdf", source / f"{name}.pdf")
        output = source.parent / "Fig1.pdf"
        shutil.copy2(build / "fig1_original_composite.pdf", output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
