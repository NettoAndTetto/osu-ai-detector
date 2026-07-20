# osu-ai-detector

`osu-ai-detector` is a local, explainable research tool for examining osu!standard beatmaps with four independent checks:

1. **Source Provenance Check** — explicit AI-use disclosures and exact matches to registered historical files.
2. **Mapperatorinator Model Agreement** — agreement with the generation process of the original Mapperatorinator checkpoints.
3. **Generator Trace Check** — mechanical traces in coordinates, timing, slider velocity, and file serialization.
4. **Mapping Structure Check** — rhythm, movement, and object-structure deviations from the calibrated human reference set.

The application deliberately reports these checks separately. It does not produce an overall verdict, and an anomaly percentile is not a probability of AI authorship.

## Supported release profile

- Windows 10/11 x64
- NVIDIA GPU with CUDA support recommended; 8 GiB VRAM minimum
- 16 GiB system RAM minimum
- About 15 GiB free space for an online installation
- About 30 GiB free space while building an offline bundle

CPU execution is supported, but Mapperatorinator Model Agreement can be extremely slow.

## Install

1. Download and extract the `v1.0.0` source release.
2. Open PowerShell in the extracted folder.
3. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

The installer is an unsigned, readable PowerShell script. It installs an isolated Miniforge environment, downloads this project's model artifacts, downloads the original Mapperatorinator source and five upstream checkpoint snapshots at pinned revisions, and verifies the recorded identities before marking the installation complete. It does not send telemetry or check for updates.

Start the local application with:

```powershell
.\start.ps1
```

Then open `http://127.0.0.1:8000`. Files are processed locally.

## Offline installation

On a connected Windows machine, run:

```powershell
.\prepare_offline_bundle.ps1 -OutputDirectory .\offline-output
```

Copy the resulting ZIP to the offline machine, extract it, and run `install-offline.ps1`. The bundle contains the application, pinned installers and package caches, the project's model artifacts, pinned upstream snapshots, and checksums. It does not contain third-party beatmaps or audio.

## Research reproduction

The application repository intentionally excludes training corpora, evaluation checkpoints, and internal recovery artifacts. Exact split identities, source revisions, hashes, acquisition/verification scripts, training code, and evaluation metadata are published separately:

- Research code: <https://github.com/NettoAndTetto/osu-ai-detector-research>
- Model artifacts: <https://huggingface.co/NettoAndTetto/osu-ai-detector-models>
- Reproducibility metadata: <https://huggingface.co/datasets/NettoAndTetto/osu-ai-detector-reproducibility>
- Methodology: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

Third-party raw beatmaps, audio, and upstream weights are not redistributed. Reproduction scripts acquire them from their original locations and verify exact registered identities.

## Responsible use

This is a forensic research aid, not proof of authorship. Review all four checks and their limitations before drawing conclusions. Do not use a single score or percentile as grounds for public accusation or punitive action.

## License

Application code is released under the MIT License. Original model artifacts and reproducibility metadata are released separately under the terms stated in their repository cards. Third-party components retain their original licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
