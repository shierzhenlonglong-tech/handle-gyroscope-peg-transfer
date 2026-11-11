# Archiving a GitHub release on Zenodo and obtaining a DOI

These steps produce a permanent, versioned snapshot of this repository with a DOI that can replace the `\todo{Zenodo DOI}` placeholder in the manuscript. They use only the official Zenodo–GitHub integration. Do not paste access tokens into chat or into this file.

1. Create a [Zenodo](https://zenodo.org/) account and, under **GitHub** in the account settings ([https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/)), authorise Zenodo to read the GitHub account that owns `handle-gyroscope-peg-transfer`.
2. On that settings page, find `shierzhenlonglong-tech/handle-gyroscope-peg-transfer` and enable the repository (the toggle should be on).
3. Upload the flattened files in this package to the GitHub repository (`main`), replacing the previous layout if needed, and confirm that scripts run from the repository root as described in `README.md`.
4. On GitHub, open **Releases** → **Draft a new release**. Create a tag such as `v1.0.0`, write a short release note (for example “Analysis scripts, frozen plan and derived tables for the BSPC manuscript”), and publish the release.
5. Zenodo will harvest that release, create an archival record, and mint a **version DOI**. A **concept DOI** that always points to the latest version is also issued. This usually completes within minutes; the new record appears in your Zenodo upload list.
6. Open the Zenodo record, check that the files, licence (MIT) and metadata are correct, and copy the DOI (for example `https://doi.org/10.5281/zenodo.xxxxxxx`). Prefer the version DOI when citing a specific snapshot in the paper; the concept DOI is useful in a running “latest version” link.
7. In `main.tex`, replace `\todo{Zenodo DOI}` with `\url{https://doi.org/10.5281/zenodo.xxxxxxx}` (or the equivalent citation), then recompile the manuscript.

If the GitHub repository is updated later, publish a new tagged release; Zenodo will mint a new version DOI under the same concept DOI.
