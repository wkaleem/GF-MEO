# GF-MEO

**GFMO: Grid Foundation Model–Embedded Optimization for AC-Feasible Unit Commitment**

This project aims to find multiperiod unit commitment decisions that satisfy
AC power flow equations and operating limits. We embed the grid foundation
model GENCO in the UC–ACOPF formulation with AC-feasible output correction
and solve the resulting formulation using augmented Lagrangian dual decomposition.

Open **[notebooks/ieee57_demo.ipynb](notebooks/ieee57_demo.ipynb)**, choose the
model size (`small` or `base`) and seed (`0` or `1`), and run the cells in order.
The notebook includes the code and IEEE57 instance and downloads the model.
Automatic setup requires Python 3.11 on Linux x86_64, internet access and `libgomp1`.

This is a prototype vibe-coded implementation under development in a personal capacity.
For collaboration, contact [wzk5140@psu.edu](mailto:wzk5140@psu.edu).
For questions, [open an issue](https://github.com/wkaleem/GF-MEO/issues).

[License](LICENSE) · [Third-party licenses](LICENSES/THIRD_PARTY_NOTICES.md)

## References

1. W. Zhang, Y. Kim, and K. Kim. [On Solving Unit Commitment with Alternating Current Optimal Power Flow on GPU](https://arxiv.org/abs/2310.13145v1), 2023.
2. D. A. Tejada-Arango et al. [Unit Commitment with ACOPF Constraints: Practical Experience with Solution Techniques](https://repositorio.comillas.edu/items/9a637581-c288-48ee-b82c-87c7594bbf9b). IEEE Milan PowerTech, 2019.
3. G. E. Constante-Flores, A. J. Conejo, and F. Qiu. [AC Network-Constrained Unit Commitment via Relaxation and Decomposition](https://experts.colorado.edu/display/pubid_505572). IEEE Transactions on Power Systems, 2022.
4. GridFM. [GENCO OPF models](https://huggingface.co/gridfm/genco-opf-datakit) and [Graphkit implementation](https://github.com/gridfm/gridfm-graphkit).
5. A. Puech et al. [GENCO — A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis](https://arxiv.org/abs/2608.09921v2), 2026.
6. S. Zeng et al. [Scalable Mixed-Integer Optimization with Neural Constraints via Dual Decomposition](https://ojs.aaai.org/index.php/AAAI/article/view/38454). AAAI, 2026.
7. A. Iftakher et al. [Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints](https://arxiv.org/abs/2507.08124). Computers & Chemical Engineering, 2026.
8. J. Nocedal and S. J. Wright. [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5). Springer, 2006.
9. O. Kröger et al. [Juniper: An Open-Source Nonlinear Branch-and-Bound Solver in Julia](https://arxiv.org/abs/1804.07332), 2018.
