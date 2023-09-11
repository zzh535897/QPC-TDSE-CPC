### QPC-TDSE-CPC
The CPC library version of QPC-TDSE[[1]](https://doi.org/10.1016/j.cpc.2023.108787). 

Stable version. 

Please note this repo is for documentation. Usual developement will NOT be made to this repo. Bug fixing and new features are tested elsewhere, and will be gathered to this repo if a new stable version is formulated.


### Issues
2023.9.11<br>
(1) The missing scaling factor sqrt(2E) in the analysis script plot_part.m has been corrected.

(2) The format of output for the real-time projection onto bound states has been corrected.

### Important
2023.9.11<br>
(1) The variable "lmd" in an output file differs a phase factor to the P_{lm}(E) defined in eq.(58):

    LMD(k,m,l)= (-i)^l \exp(i\Delta_{l}(k)) P_{lm}(E)

This will not affect the use of eq.(65) but could cause some confusion when calculating physical quantities such as the time-delay.

(2) The gaussian envelope in the original release is really "gaussian", which means it is non-zero everywhere. The simulation starts and ends at 2FWHM far from its central peak. If the PCS method is applied for this case, some unphysical results would occur since the instantaneous A(t_f) is non-zero. 

A modified gaussian envelope with its tail cut should be customized instead. One may refer to include/structure/field.h for its format.

### References
[1] Zhao-Han Zhang, Yang Li, Yi-Jia Mao, Feng He,
*QPC-TDSE: A parallel TDSE solver for atoms and small molecules in strong lasers*
[Comput. Phys. Comm., **290** 108787 (2023)]
(https://doi.org/10.1016/j.cpc.2023.108787)
