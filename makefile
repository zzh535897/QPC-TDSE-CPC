include ./env/path.mk
include ./env/make_common.mk

sphTDSE:$(QPC_PATH)/bin/sphTDSE
	$(call Author)

%  :$(QPC_PATH)/bin/%
	$(call Author)

$(QPC_PATH)/bin/%  :$(addprefix $(QPC_PATH)/bin/,%.o)
	$(call Linking,$@)
	@mkdir -p $(QPC_PATH)/bin
	@mkdir -p $(LOG)
	@$(GCC) $(ALL_LDFLAGS) $(DEBUG) -o $@    $+ 2>$(LD_LOG)

$(QPC_PATH)/bin/%.o: $(addprefix $(QPC_PATH)/config/,%.cpp) $(QPC_PATH)/binsrc/sphTDSE.cpp
	$(call Compile,$@)
	@mkdir -p $(QPC_PATH)/bin
	@mkdir -p $(LOG)
	@$(GCC) -D PARA_PATH=\"$(QPC_PATH)/config/$*.cpp\" $(ALL_CCFLAGS) $(DEBUG) -o $@ -c $(QPC_PATH)/binsrc/sphTDSE.cpp 2>$(CC_LOG)

.PRECIOUS:$(QPC_PATH)/bin/%

.PHONY:clean cleanbin cleanlog cleanobj

clean:cleanbin cleanlog cleanobj

cleanbin:
	rm -f $(wildcard $(QPC_PATH)/bin/*);

cleanlog:
	rm -f $(wildcard $(QPC_PATH)/log/*.log);

cleanobj:
	rm -f $(wildcard $(QPC_PATH)/bin/*.o);
