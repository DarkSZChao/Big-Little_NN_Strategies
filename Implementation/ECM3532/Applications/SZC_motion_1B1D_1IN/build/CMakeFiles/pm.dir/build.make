# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/936/bin/cmake

# The command to remove a file.
RM = /snap/cmake/936/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build

# Include any dependencies generated for this target.
include CMakeFiles/pm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pm.dir/flags.make

../SZC_motion_energy_measurement.ld: ../config.h
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../SZC_motion_energy_measurement.ld"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/linker/flash_shm_relocate_to_ram.ld.S -P -o /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/SZC_motion_energy_measurement.ld -I /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement

../config.h:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "genconfig file"
	/snap/cmake/936/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/.config /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/genconfig.py --header-path /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/Kconfig

CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o: CMakeFiles/pm.dir/flags.make
CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c
CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o: CMakeFiles/pm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o -MF CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o.d -o CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c

CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c > CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.i

CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c -o CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.s

CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o: CMakeFiles/pm.dir/flags.make
CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c
CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o: CMakeFiles/pm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o -MF CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o.d -o CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c

CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c > CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.i

CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c -o CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.s

# Object files for target pm
pm_OBJECTS = \
"CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o" \
"CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o"

# External object files for target pm
pm_EXTERNAL_OBJECTS =

/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libpm.a: CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/ecm3532/src/eta_pwr_gov.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libpm.a: CMakeFiles/pm.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/hw/power/common/src/dvfs_mon.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libpm.a: CMakeFiles/pm.dir/build.make
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libpm.a: CMakeFiles/pm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking C static library /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libpm.a"
	$(CMAKE_COMMAND) -P CMakeFiles/pm.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pm.dir/build: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libpm.a
.PHONY : CMakeFiles/pm.dir/build

CMakeFiles/pm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pm.dir/clean

CMakeFiles/pm.dir/depend: ../SZC_motion_energy_measurement.ld
CMakeFiles/pm.dir/depend: ../config.h
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles/pm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pm.dir/depend

