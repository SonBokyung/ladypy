# LadyPy

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)

**LadyPy** is Python Library for simulating _Language Dynamics models_.

## Dependencies

The package depend on [`numpy`](http://www.numpy.org/), the numerical library for Python,
and [`tqdm`](https://github.com/tqdm/tqdm) for an elegant progress bar support.

## How to use

### Install

```bash
make install
# or
pip install .
```

For an editable install, you can use commands as below:

```bash
make install-symlink
# or
pip install -e .
```

### Uninstall

```bash
make uninstall
# or
make remove
# or
pip uninstall ladypy
```

## Developments

Read the following description: [Link](./development.md).

## Reference

- Nowak, M. A., Plotkin, J. B., & Krakauer, D. C. (1999). The evolutionary language game. Journal of Theoretical Biology, 200(2), 147-162. http://doi.org/10.1006/jtbi.1999.0981
- Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics. Reviews of modern physics, 81(2), 591.
