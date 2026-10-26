# Acknowledgements

This repository is implemented on top of the public [UniRank](https://github.com/salmon1802/UniRank) benchmark (Apache License 2.0). HSAF reuses UniRank's unified data processing, blocked Parquet loading, feature-map construction, distributed training loop, multi-task evaluation, and several baseline implementations. We thank the UniRank authors for making this infrastructure publicly available.

The preprocessing scripts and dataset schemas in `data_preparation/` and `docs/dataset.md` are adapted from UniRank. The baselines in `model_zoo/` are also derived from the corresponding UniRank implementations. The HSAF additions are exposed through `model_zoo/HSAF.py`; the exact numerical core is retained in `model_zoo/EviTime_gatecalib_use_time.py` to preserve reproducibility.

If you use this code, please cite both HSAF and UniRank:

```bibtex
@misc{li2026unirank,
  author = {Li, Honghao and Wang, Xianquan and Zhang, Zibin and Zhang, Yi and Lin, Kangyi and Zhang, Yiwen},
  title = {{UniRank}: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction},
  year = {2026},
  howpublished = {arXiv:2607.19987}
}
```
