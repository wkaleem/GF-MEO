# GF-MEO

**Grid Foundation Model–Embedded Optimization for AC-Feasible Unit Commitment**

Joint optimization of commitment decisions and GENCO inputs, with local AC
output correction and multiperiod coordination.

Open **[notebooks/ieee57_demo.ipynb](notebooks/ieee57_demo.ipynb)** and run the cells in order.
Choose **model size** (`small` or `base`) and **seed** (`0` or `1`).
The notebook includes the source and case and downloads the matching checkpoint.

- IEEE57, four hourly periods, one switchable generator, 300-second CPU budget.
- Automatic setup: Python 3.11, Linux x86_64, internet access and `libgomp1`.
- A compatible interpreter can be selected through `GFMO_DEMO_PYTHON`.
- Fresh solutions are checked against the original UC–ACOPF constraints.

This is a constructed demonstration, not held-out evaluation. Local correction
does not guarantee a feasible or globally optimal solution.

Original code: **[Research Evaluation License](LICENSE)**.
[Third-party terms](THIRD_PARTY_NOTICES.md) remain applicable.

Waquar Kaleem · The Pennsylvania State University · **wzk5140@psu.edu**

AI-assisted development and documentation.
