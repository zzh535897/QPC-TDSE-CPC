#!/bin/bash 

targetname="example01_a_gs";

load_path="0";
save_path="./example01_a_gs.h5";

paramL="@L $load_path $save_path";

parama="@a 0 0 0 0 o"; #(is, im, il, in, re, im)

make -B $targetname && ./bin/$targetname $paramL $parama



