# Scientific source archive

Training, preprocessing, model implementations, geometry, selection and checking scripts retained from the research archive. Code is MIT licensed. Original and public file hashes are recorded in `../provenance/source_mapping.json`.

These scripts are an inspectable archive, not a one-command training package. Directory names reflect their archival grouping: historical imports and `ROOT` paths expected modules in a campaign directory, with `fixed_comparison` under `confirmation_v1`. Training additionally requires the prepared local_data_v2 caches, split locks, pretrained weights and configuration records. Those caches and weights are not included in this release. Do not disable hash checks to force execution or interpret historical acceptance gates as certification of this edited release.

The raw 12 source files are supplied separately in Releases. `recovered_preparation/prepare_local_original.py` is the retained original preprocessing implementation; the later prepare_local.py may intentionally guard against rerunning the original preparation. Rebuild in a new work directory and retain new receipts instead of replacing historical records.

Dependencies vary by script (NumPy, PyTorch, timm, Pillow, SciPy and geometry/image packages). No complete training environment or clean-machine training reproduction is claimed. The directly exercised public workflow is the result-table analysis in `../analysis`.

Internal names such as `confirmation`, `sealed` and `freeze` describe computational stages. The study was developed using these data; evaluation annotations had been decoded and cached during original preparation. These names do not establish preregistration or a wholly unseen test.
