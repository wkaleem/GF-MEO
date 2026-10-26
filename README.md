# GF-MEO

**Grid Foundation Model–Embedded Optimization for AC-Feasible Unit Commitment**

This project aims to find multiperiod unit commitment decisions that satisfy
AC power flow equations and operating limits. We embed the grid foundation
model GENCO in the UC–ACOPF formulation with AC-feasible output correction
and solve the resulting formulation using augmented Lagrangian dual decomposition.

Open **[notebooks/ieee57_demo.ipynb](notebooks/ieee57_demo.ipynb)**, choose the
model size (`small` or `base`) and seed (`0` or `1`), and run the cells in order.
The notebook includes the code and IEEE57 instance and downloads the model.
Automatic setup requires Python 3.11 on Linux x86_64, internet access and `libgomp1`.

This is a prototype implementation, and work is in progress.
For questions, [open an issue](https://github.com/wkaleem/GF-MEO/issues).

[License](LICENSE) · [Third-party licenses](THIRD_PARTY_NOTICES.md)
