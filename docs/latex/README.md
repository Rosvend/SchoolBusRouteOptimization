# LaTeX Setup Guide for VS Code

This guide explains how to set up your environment to compile and edit the LaTeX documents in this project using Visual Studio Code. It covers instructions for both Linux (Ubuntu) and Windows.

## Prerequisites

### 1. Install a LaTeX Distribution

**For Linux (Ubuntu 24.04 LTS and similar):**
Open your terminal and run the following command to install TeX Live. We recommend `texlive-full` to avoid missing packages:
```bash
sudo apt update
sudo apt install texlive-full
```
*(Note: `texlive-full` is quite large. If you prefer a smaller installation, you can install `texlive-latex-base` and `texlive-latex-extra` instead, but you might need to install additional packages manually later).*

**For Windows:**
We recommend using **MiKTeX**, as it can automatically download missing packages on the fly.
1. Download the MiKTeX installer from [https://miktex.org/download](https://miktex.org/download).
2. Run the installer and follow the standard setup wizard.
3. **Important:** When prompted for "Install missing packages on-the-fly", choose "Yes".

Alternatively, you can install [TeX Live for Windows](https://tug.org/texlive/windows.html).

### 2. Configure Visual Studio Code

1. Open Visual Studio Code.
2. Go to the Extensions view (`Ctrl+Shift+X` on Windows/Linux).
3. Search for and install the **LaTeX Workshop** extension (by *James Yu*).

## Compiling the Document

1. Open the LaTeX source file (`docs/latex/optimization.tex`) in VS Code.
2. The LaTeX Workshop extension should automatically detect the `.tex` file.
3. Save the file (`Ctrl+S`) or press `Ctrl+Alt+B` to build the LaTeX project. It typically uses `latexmk` under the hood.
4. To view the compiled PDF right inside VS Code, press `Ctrl+Alt+V` or click the "View PDF" icon in the top right corner. The PDF will automatically update every time you build.

## Troubleshooting

- **Windows MiKTeX Package Prompts**: During the first build, MiKTeX might show popup windows asking to install missing packages. Accept them to proceed. If it hangs in VS Code, you might need to compile it manually in the terminal (`pdflatex optimization.tex`) the first time to click "Accept".
- **Commands Not Found**: If VS Code complains that `pdflatex` or `latexmk` is not found, ensure that the path to your LaTeX binaries is added to your system's `PATH` environment variable. (The Ubuntu package manager and MiKTeX installer usually handle this automatically, but sometimes a reboot or VS Code restart is required).
