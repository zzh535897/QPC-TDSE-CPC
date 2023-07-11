#!/bin/bash 

targetname="example03_a";
make -B $targetname

A01="1.6219";
Om1="0.0570";
nT1="16";
Sh1="0";
Ce1="0";

paramZ=$paramZ" @F Z00 $A01 Z01 $Om1 Z02 $nT1 Z03 $Sh1 Z04 $Ce1"; #(F0, w0, nT, n0, cep)

load_path="./example03_a_gs_R2.h5";
save_path="./example03_a_R2.h5";

paramP="@P0 1 1 0 0 1 1 0"; #(Rp,Z1,TH1,PH1,Z2,TH2,PH2)
paramO="@O $load_path $save_path";

./bin/$targetname $paramP $paramO $paramZ

load_path="./example03_a_gs_R3.h5";
save_path="./example03_a_R3.h5";

paramP="@P0 1.5 1 0 0 1 1 0"; #(Rp,Z1,TH1,PH1,Z2,TH2,PH2)
paramO="@O $load_path $save_path";

./bin/$targetname $paramP $paramO $paramZ
