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

The installer is an unsigned, readable PowerShell script. It installs an isolated Miniforge environment, downloads this project's model artifacts and five upstream checkpoint snapshots at pinned revisions, and verifies the recorded identities before marking the installation complete. The pinned Mapperatorinator source code is already bundled with the release, so it is not fetched during installation. The installer does not send telemetry or check for updates.

### Manual model download fallback

The installer makes one automatic model-download attempt. If Hugging Face is unavailable or unreliable, do not keep retrying it. Download only the missing repositories in a browser and place their files under:

```text
manual-models/
  osu-ai-detector-models/
  v29/
  v30/
  v31/
  v32/
  v32-mini/
```

Use these exact pinned pages:

- [osu-ai-detector-models v1.0.0](https://huggingface.co/NettoAndTetto/osu-ai-detector-models/tree/v1.0.0)
- [Mapperatorinator v29.1](https://huggingface.co/OliBomby/Mapperatorinator-v29.1/tree/656db0cd04a8a6a77d94a96e7af89810fb6de5ef)
- [Mapperatorinator v30](https://huggingface.co/OliBomby/Mapperatorinator-v30/tree/a4c6e6e69c055711c2293d63161c0e52980e56a1)
- [Mapperatorinator v31](https://huggingface.co/OliBomby/Mapperatorinator-v31/tree/12772791b862b97a11153aa766b2481afa5dda11)
- [Mapperatorinator v32](https://huggingface.co/OliBomby/Mapperatorinator-v32/tree/74f22583400d259bf424819e11027c17933efe54)
- [Mapperatorinator v32-mini](https://huggingface.co/OliBomby/Mapperatorinator-v32-mini/tree/7807f0dc70cab671be012e1f5ddf945b0b8b7278)

Preserve the repository directory structure while downloading. For v29–v31 this includes root-level `config.json`, `tokenizer.json`, and the complete `model.safetensors`; v32 and v32-mini place the required files under `gamemode=0/`. A completed repository already present in the automatic-download cache may be omitted. Then run:

```powershell
.\install.ps1 -ManualModelsDirectory .\manual-models
```

The installer imports these ordinary files into the same pinned local cache used by inference and verifies every required file against the release's recorded byte size and SHA-256. It does not switch the detector to a less strict model-loading path.

If Conda package retrieval is blocked rather than model retrieval, prepare the fully offline bundle on another connected Windows machine as described below. Manually assembling hundreds of Conda packages is not supported.

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
