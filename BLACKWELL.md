# ColabFold on Blackwell (sm_120) — IPI internal fork

This branch (`blackwell`) is **ColabFold `v1.6.1`** plus the single change required to run
on NVIDIA Blackwell / RTX PRO 6000 (`sm_120`) with a modern JAX (>= 0.10, which ships the
`sm_120` kernels). It is used on the IPI **gnode2** workstation.

## Why base this on the `v1.6.1` release tag (not `main`)

`sokrypton/ColabFold@main` and its companion `sokrypton/alphafold@main` have **drifted into an
inconsistent pair**: colabfold `main` calls `RunModel(extended_ptm_config=...)`, but the alphafold
`main` tree does not accept that argument (it lacks the `calc_extended_ptm` code). Installing both
mains therefore fails at load time:

```
TypeError: RunModel.__init__() got an unexpected keyword argument 'extended_ptm_config'
```

The **released** pair — `colabfold 1.6.1` + `alphafold-colabfold 2.3.13` — is internally
consistent and tested, so this fork is based on the `v1.6.1` tag. The matching alphafold is the
companion fork [`proteininnovation/alphafold@blackwell`](https://github.com/proteininnovation/alphafold/tree/blackwell)
(the 2.3.13 source, which upstream only ever shipped to PyPI, vendored + patched — see its
`BLACKWELL.md`).

## The change (1 file)

`colabfold/colabfold.py` — `clear_mem()` used `jax.lib.xla_bridge`, which was **removed in
JAX >= 0.10**. Replaced with the modern `jax.live_arrays()` API (with a graceful fallback):

```python
def clear_mem(device="gpu"):
  try:
    for arr in jax.live_arrays(device): arr.delete()
  except Exception:
    try:
      backend = jax.extend.backend.get_backend(device)
      for buf in backend.live_buffers(): buf.delete()
    except Exception:
      pass
```

That is the only functional edit; everything else is stock `v1.6.1`.

## Install (as used on gnode2)

Install the companion alphafold fork **first** so it satisfies colabfold's pinned
`alphafold-colabfold == 2.3.13` dependency (no PyPI pull), then this fork:

```bash
pip install "jax[cuda12]"                                                   # sm_120 kernels
pip install "git+https://github.com/proteininnovation/alphafold@blackwell" # patched 2.3.13
pip install "git+https://github.com/proteininnovation/ColabFold@blackwell" # this fork (1.6.1 + clear_mem)
```

## Scope / provenance

- Internal fork for IPI hardware. **Not** submitted upstream (by choice).
- Related upstream reports: sokrypton/ColabFold #813, #841.
- This branch is force-published; it intentionally does **not** track upstream `main`.
