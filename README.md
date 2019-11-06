# Autocorrelation mosaicing via OMP

The autocorrelation mosaicing code!

This is included as a small gesture towards basic reproducibility and can indeed reproduce the results of the paper.
I am rewriting it to be faster, because, well, it's too slow right now.

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
* [Proceeedings](https://ismir2019.ewi.tudelft.nl/?q=accepted-papers#session_G)

You can cite this using BibTeX. ([download](./paper.bib)).

```bibtex
@inproceedings{MacKinlayMosaic2019,
  address = {Delft},
  author = {MacKinlay, Daniel},
  booktitle = {Proceedings of {ISMIRd}},
  language = {en},
  pages = {5},
  title = {Mosaic {Style Transfer} Using {Sparse Autocorrelograms}},
  year = {2019}
}
```

<script type="text/javascript">
  var _gauges = _gauges || [];
  (function() {
    var t   = document.createElement('script');
    t.type  = 'text/javascript';
    t.async = true;
    t.id    = 'gauges-tracker';
    t.setAttribute('data-site-id', '57847ffcc88d9002a400fd75');
    t.setAttribute('data-track-path', 'https://track.gaug.es/track.gif');
    t.src = 'https://d2fuc4clr7gvcn.cloudfront.net/track.js';
    var s = document.getElementsByTagName('script')[0];
    s.parentNode.insertBefore(t, s);
  })();
</script>