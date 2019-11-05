# Autocorrelation mosaicing via OMP

Welcome to a messy prototype of the autocorrelation mosaicing code.

This is included as a small gesture towards basic reproducibility and can indeed reproduce the results of the paper.
I am rewriting it to be faster, because, well, it's too slow right now - in particular, slower than real-time.

Install it by running

```shell
pip install -r requirements.txt
pip install -e .
jupyter notebook render_examples.ipynb
```

Persistent links:

* [Examples](https://danmackinlay.github.io/mosaicing_omp_ismir_2019/examples/demo_autocorr/index_autocorr.html).
* [Source](https://github.com/danmackinlay/mosaicing_omp_ismir_2019)
* [preprint of the ISMIR paper](./mosaicing_omp_2019.pdf).
* [Proceeeedings](https://ismir2019.ewi.tudelft.nl/?q=accepted-papers#session_G)

You can cite this using BibTeX. ([download](./paper.bib)).

```bibtex
@inproceedings{MacKinlayMosaic2019,
  address = {{Delft}},
  author = {MacKinlay, Daniel},
  booktitle = {Proceedings of {{ISMIR}}},
  copyright = {All rights reserved},
  language = {en},
  pages = {5},
  title = {Mosaic {{Style Transfer}} Using {{Sparse Autocorrelograms}}},
  year = {2019}
}
```
