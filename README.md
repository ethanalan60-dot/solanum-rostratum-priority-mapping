# Solanum rostratum priority mapping

Original imagery, patch annotations and scientific source code for studying how image detail and grid size affect priority-area selection under area limits.

**One scene, four adjacent tiles.** This is a single-scene computational study, not independent multi-field validation. See [research scope and AI assistance](provenance/RESEARCH_SCOPE.md).

## Download the complete source data

[Release v1.0.0](https://github.com/ethanalan60-dot/solanum-rostratum-priority-mapping/releases/tag/v1.0.0) contains four archives, one per tile (`4_5`–`4_8`). Together they contain all 12 supplied TIFF, indexed PNG-mask and JSON-annotation files (1,448,465,956 uncompressed bytes). JSON files also retain their embedded image data. No resizing, synthetic replacement or re-annotation was applied for publication.

Download and verify all four archives with Python 3.10 or newer:

```sh
python tools/download_data.py --output downloads
```

[File sizes and SHA-256 hashes](data/manifest.json) allow byte-level verification. Each ZIP also contains a SHA256.json file. Extract the ZIPs using your archive utility. Original image coordinates/georeferencing are retained.

## Reproduce the grid-origin summaries and figure

```sh
python -m pip install -r requirements-analysis.txt
python analysis/summarize_and_plot.py
python tools/verify_results.py
```

This regenerates summary tables and Figure S18 from every one of the 4,608 retained result rows. For Chinese plots, install Microsoft YaHei or Noto Sans CJK SC; English plots use DejaVu Sans. It does not rerun model inference or verify raw-to-prediction equivalence. Full training, model and geometry implementations are provided under [source/](source/README.md); trained weights and prepared caches are not included, so end-to-end training reproduction is not claimed.

## Licensing

- Study code: [MIT](LICENSE).
- Original local imagery, annotations and study-derived tables: [CC BY 4.0](data/LICENSE.md).
- Dependencies and external data retain their respective licenses.

The data rights holder authorized complete public download and reuse. Preserve attribution and state modifications. Suggested dataset citation: **ethanalan60-dot (2026). Solanum rostratum priority mapping dataset and code, v1.0.0. GitHub.** A manuscript by Xiao Zhang, Fenting Wang, Feng Qin, Lijuan Yin, Wanqiang Qian and Xi Qiao is associated with this work; no journal acceptance or DOI is asserted.

## 中文说明

本仓库公开一处场景的四块相邻原始影像及标注，共12个文件，并提供研究源码及全部4,608组网格原点分析结果。标注表示刺萼龙葵粗略斑块范围，不能直接解释为纯叶片覆盖率或实际防除效果。完整数据通过 Releases 下载，代码采用 MIT，数据采用 CC BY 4.0。

公开整理清除了私人电脑路径和无关管理文件，保留真实的 AI 辅助说明、数据局限及文件校验记录。已有结果表可直接复算统计与图；完整训练还需要整理后的缓存、权重和运行配置，本版本不声称一键复现全部训练。
