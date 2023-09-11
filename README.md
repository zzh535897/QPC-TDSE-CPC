# QPC-TDSE-CPC
The CPC library version of QPC-TDSE[[1]](https://doi.org/10.1016/j.cpc.2023.108787). 

Stable version. 

Please note this repo is for documentation. Usual developement will NOT be made to this repo. Bug fixing and new features are tested elsewhere, and will be gathered to this repo if a new stable version is formulated.


## Issues
2023.9.11<br>
(1) The missing scaling factor sqrt(2E) in the analysis script plot_part.m has been corrected. 

(2) The format of output "proj_val" for the real-time projection onto bound states has been corrected. Now it follows the format in the user guide.

## Important
2023.9.11<br>
(1) The variable "lmd1" in an output file actually differs a phase factor to the P_{lm}(E) as

    LMD(k,m,l)= (-i)^l \exp(i\Delta_{l}(k)) P_{lm}(E)

Here P_{lm}(E) is defined in eq.(58). This feature will not affect the use of eq.(65), but could cause confusion when calculating physical quantities such as the time-delay.

(2) The gaussian envelope in the original release is really "gaussian", which means it is non-zero everywhere. The simulation starts and ends at 2FWHM far from its central peak. If the PCS method is applied for this case, some unphysical results would occur since the instantaneous A(t_f) is non-zero. 

A modified gaussian envelope with its tail cut should be customized instead. One may refer to include/structure/field.h for its format.

(3) The projection onto field-free eigenstates could differ a sign between two runs using different box size parameters in the paper version code, since the LAPACK diagonalization routines randomly choose the sign of the eigenvectors. This may cause some confusion.

In the latest version we always fix the sign of eigen states to be positive near the origin.

## References
[1] Zhao-Han Zhang, Yang Li, Yi-Jia Mao, Feng He,
*QPC-TDSE: A parallel TDSE solver for atoms and small molecules in strong lasers*
[Comput. Phys. Comm., **290** 108787 (2023)]
(https://doi.org/10.1016/j.cpc.2023.108787)
