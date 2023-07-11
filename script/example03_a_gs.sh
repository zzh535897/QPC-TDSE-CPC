#!/bin/bash 

targetname="example03_a_gs";
make -B $targetname

paramP="@P0 1.0 1 0 0 1 1 0"; #(Rp,Z1,TH1,PH1,Z2,TH2,PH2)
paramA="@A 0 -1.15 -1.05 0 o"; #(is, El, Eh, id, re, im), about -1.1

load_path="0";
save_path="./example03_a_gs_R2.h5";

paramO="@O $load_path $save_path";
./bin/$targetname $paramP $paramO $paramA

paramP="@P0 1.5 1 0 0 1 1 0"; #(Rp,Z1,TH1,PH1,Z2,TH2,PH2)
paramA="@A 0 -0.95 -0.85 0 o"; #(is, El, Eh, id, re, im), about -0.9

load_path="0";
save_path="./example03_a_gs_R3.h5";

paramO="@O $load_path $save_path";
./bin/$targetname $paramP $paramO $paramA
