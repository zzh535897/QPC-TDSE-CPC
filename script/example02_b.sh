#!/bin/bash 

targetname="example02_b";

A01="1.675122512372942"; 
Om1="0.057";
nT1="6";
Sh1="0";
Ce1="0.0"; 	

A02="-1.256341884279707"; 
Om2="0.057";
nT2="6";
Sh2="0";
Ce2="0.5";	

paramX=$paramX" @F X00 $A01 X01 $Om1 X02 $nT1 X03 $Sh1 X04 $Ce1"; #(F0, w0, nT, n0, cep)
paramY=$paramY" @F Y00 $A02 Y01 $Om2 Y02 $nT2 Y03 $Sh2 Y04 $Ce2"; #(F0, w0, nT, n0, cep)

parama="@a 0 z 0 0 o";	#(is, im, il, in, re, im), character 'z' means let m=zero

load_path="0";
save_path="./example02_b.h5";

paramL="@E $load_path $save_path"; #use @E to call the EP programme

make -B $targetname && ./bin/$targetname $paramL $paramX $paramY $parama



