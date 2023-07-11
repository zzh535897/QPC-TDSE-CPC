#!/bin/bash 

targetname="example03_b_gs";


paramP="@P0 1.0 1 0 0 1 1 0";

#(is, El, Eh, id, re, im)
paramA="@A 0 -1.12 -1.08 0 o";

load_path="./example03_b_gs.h5";
save_path="./example03_b_gs.h5";

paramO="@O $load_path $save_path";

make -B $targetname && ./bin/$targetname $paramP $paramO $paramA



