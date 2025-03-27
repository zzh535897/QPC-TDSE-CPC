
GCC		:= $(GCC_PATH)/bin/g++

INC		:= -I $(QPC_PATH)/include
INC		+= -I $(MKL_PATH)/include
INC		+= -I $(GSL_PATH)/include
INC		+= -I $(HDF_PATH)/include

LIB		:= -L $(QPC_PATH)/lib
LIB		+= -L $(GCC_PATH)/lib64
LIB		+= -L $(MKL_PATH)/lib/intel64
LIB		+= -L $(ICX_PATH)/lib
LIB 		+= -L $(GSL_PATH)/lib
LIB		+= -L $(HDF_PATH)/lib
########################################################################

#use -fopt-info-vec to show vectorization info only.
#use -fopt-info-vec-missed to show why vectorization is not used.
CCFLAGS 	:= -O3
CCFLAGS		+= -fopt-info --param vect-max-version-for-alias-checks=1000 -funroll-loops -ffunction-sections -fdata-sections -finline-functions -fexpensive-optimizations #-fprefetch-loop-arrays #-ffast-math
CCFLAGS		+= -fopenmp 
CCFLAGS		+= -Wall 
CCFLAGS		+= -std=c++17
CCFLAGS		+= -pedantic 
CCFLAGS		+= -march=native -m64 -lmvec -fomit-frame-pointer -mveclibabi=svml
CCFLAGS		+= -DMKL_ILP64

########################################################################
LDFLAGS		:= -fopenmp 
LDFLAGS		+= -Wl,--gc-sections -Wl,--no-as-needed
LDFLAGS		+= -Wl,-rpath=$(QPC_PATH)/lib
LDFLAGS         += -Wl,-rpath=$(GCC_PATH)/lib64
LDFLAGS		+= -Wl,-rpath=$(MKL_PATH)/lib/intel64 -lmkl_intel_ilp64 -lmkl_rt -lmkl_intel_thread -lmkl_core 
LDFLAGS		+= -Wl,-rpath=$(ICX_PATH)/lib -liomp5 -lpthread -lm -ldl
LDFLAGS	 	+= -Wl,-rpath=$(GSL_PATH)/lib -lgsl
LDFLAGS 	+= -Wl,-rpath=$(HDF_PATH)/lib -lhdf5 -ldl -lz -lm

########################################################################
DEBUG		:= #-pg

ALL_CCFLAGS	:=$(CCFLAGS)
ALL_CCFLAGS	+=$(INC)

ALL_LDFLAGS	:=$(LDFLAGS)
ALL_LDFLAGS	+=$(LIB)

########################################################################
LOG		?= $(QPC_PATH)/log

LD_LOG 		?= $(QPC_PATH)/log/ld_error.log
CC_LOG 		?= $(QPC_PATH)/log/cc_error.log
########################################################################

define Author
	@echo -e '\e[36m'
	@echo -e '#######################################'
	@echo -e
	@echo -e '########### QPC-TDSE v1.0 #############'
	@echo -e
	@echo -e '#######################################'
	@echo -e
	@echo -e '############# 2023-03-24 ##############'
	@echo -e
	@echo -e ' Copyright(C) 2022-2023 Zhao-Han Zhang '
	@echo -e
	@echo -e '#### mail:zhangzhaohan@sjtu.edu.cn ####'
	@echo -e
	@echo -e '#######################################'
	@echo -e '\e[0m'

endef

define Compile
	@echo -e '\e[32m COMPILING \e[31m '$(1)'\e[0m'

endef

define Linking
	@echo -e '\e[32m LINKING   \e[31m '$(1)'\e[0m'
endef

