#!/bin/bash 

targetname="example01_b";

A01="0.9364";
Om1="0.0570";
nT1="6";
Sh1="0";
Ce1="0";

paramZ=$paramZ" @F Z00 $A01 Z01 $Om1 Z02 $nT1 Z03 $Sh1 Z04 $Ce1"; #(F0, w0, nT, n0, cep)

parama="@a 0 0 0 0 o"; #(is, im, il, in, re, im)

load_path="0";
save_path="./example01_b.h5";

paramL="@L $load_path $save_path";

make -B $targetname && ./bin/$targetname $paramL $paramZ $parama



