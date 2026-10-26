# Third-party materials

The evaluation-only license covers original GFMO materials. It does not
relicense or restrict the following dependencies, models or data.

- **GENCO / OpenGridFM Graphkit**, Apache-2.0. The notebook downloads the
  unmodified source at commit `4e7fbcf6aa33c09b5ec9f15e5952dbbff004b74e`
  from <https://github.com/gridfm/gridfm-graphkit> and preserves its `LICENSE`.
- **Released GENCO OPF models**, Apache-2.0, downloaded from
  <https://huggingface.co/gridfm/genco-opf-datakit> at revision
  `2d2401572f6a2d0c1be79841b2def99cf45e1639`. Configuration, weights and
  normalization statistics are matched and SHA-256 checked.
- **OPF scenario data**, Apache-2.0, from
  <https://huggingface.co/datasets/gridfm/opf_small_case57_ieee> at revision
  `59d1d1ae81c5bdf2c73ec8216c5fb8fbf93ae64a`. `data/ieee57_uc.json` derives
  from scenario 308, load group 13, with a four-period demand profile and
  additional commitment, cost, duration, ramp and initial-condition data.
  Electrical equipment and generation-cost coefficients are unchanged.
- **OpenGridFM Datakit**, Apache-2.0, supplies the corrected IEEE57 nominal
  case at commit `316c4afeb1b1d048b6ccaa6135fbc54e00549f64`:
  <https://github.com/gridfm/gridfm-datakit>.
- **PGLib-OPF**, IEEE PES Task Force on Benchmarks for Validation of Emerging
  Power System Algorithms: underlying network data are CC BY 4.0, software MIT.
  See <https://github.com/power-grid-lib/pglib-opf> and its license. The public
  case is an adapted operating scenario and UC extension, not an unmodified
  PGLib benchmark. No endorsement by the original authors is implied.

The [Apache-2.0 license text](LICENSES/Apache-2.0.txt) is bundled with this
release. The underlying data attribution uses
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

Runtime dependencies retain their respective licenses. They are installed from
their official distributions rather than relicensed under this repository.
The local AC correction follows a constrained nearest-point/KKT formulation;
the implementation here is original GFMO code, not a redistributed KKT-hardnet
implementation. Related work: <https://github.com/SOULS-TAMU/kkt-hardnet>.
