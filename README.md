# Autocorrelation mosaicing via OMP

Welcome to a messy prototype of the autocorrelation mosaicing code.

This is included as a small gesture towards basic reproducibility and can indeed reproduce the results of the paper.
I am rewriting it in julia to be faster, because it's much slower than real-time at the moment.

Install it by running

```shell
pip install -r requirements.txt
pip install -e .
jupyter notebook render_examples.ipynb
```

The preprint of the final ISMIR paper is available
[in this repository](./mosaicing_omp_2019.pdf).

[Examples are here](./examples/demo_autocorr/index_autocorr.html).