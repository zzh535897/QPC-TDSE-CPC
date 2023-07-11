#!/bin/bash 

targetname="example01_c";

A01="1.8728";
Om1="0.0570";
nT1="6";
Sh1="0";
Ce1="0";

paramZ=$paramZ" @F Z00 $A01 Z01 $Om1 Z02 $nT1 Z03 $Sh1 Z04 $Ce1"; #(F0, w0, nT, n0, cep)

load_path="./example01_c_gs.h5";
save_path="./example01_c.h5";

paramL="@L $load_path $save_path";

make -B $targetname && ./bin/$targetname $paramL $paramZ $parama



