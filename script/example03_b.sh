#!/bin/bash 

targetname="example03_b";
make -B $targetname;

paramP="@P0 1 1 0 0 1 1 0"; #(R, Z1, TH1, PH1, Z2, TH2, PH2)

A01_list=("0.0" "0.07071068" "0.1");
A02_list=("0.1" "0.07071068" "0.0");
A03_list=("0.1" "0.1"        "0.1");
save_list=("./example03_b_CPdeg000.h5" "./example03_b_CPdeg045.h5" "./example03_b_CPdeg090.h5");

for((i=2;i<3;++i));
do
	
	load_path="./example03_b_gs.h5";
	save_path=${save_list[$i]};

	A01=${A01_list[$i]};
	Om1="2.0";
	nT1="4.0";
	Sh1="0.0";
	Ce1="0.0";

	A02=${A02_list[$i]};
	Om2="2.0";
	nT2="4.0";
	Sh2="0.0";
	Ce2="0.0";

	A03="0.1";
	Om3="2.0";
	nT3="4.0";
	Sh3="0.0";
	Ce3="0.5";

	paramF=$paramF" @F Z00 $A01 Z01 $Om1 Z02 $nT1 Z03 $Sh1 Z04 $Ce1"; #(F0, w0, nT, n0, cep)
	paramF=$paramF" @F X00 $A02 X01 $Om2 X02 $nT2 X03 $Sh2 X04 $Ce2"; #(F0, w0, nT, n0, cep)
	paramF=$paramF" @F Y00 $A03 Y01 $Om3 Y02 $nT3 Y03 $Sh3 Y04 $Ce3"; #(F0, w0, nT, n0, cep)

	paramO="@O $load_path $save_path";

	./bin/$targetname $paramP $paramO $paramF
done
