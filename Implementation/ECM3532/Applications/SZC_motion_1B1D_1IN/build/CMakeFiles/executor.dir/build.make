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
CMAKE_COMMAND = /snap/cmake/955/bin/cmake

# The command to remove a file.
RM = /snap/cmake/955/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build

# Include any dependencies generated for this target.
include CMakeFiles/executor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/executor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/executor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/executor.dir/flags.make

../SZC_motion_1B1D_1IN.ld: ../config.h
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../SZC_motion_1B1D_1IN.ld"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/linker/flash_shm_relocate_to_ram.ld.S -P -o /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/SZC_motion_1B1D_1IN.ld -I /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN

../config.h:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "genconfig file"
	/snap/cmake/955/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/.config /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/genconfig.py --header-path /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/Kconfig

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o: CMakeFiles/executor.dir/flags.make
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o: CMakeFiles/executor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o -MF CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o.d -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c > CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.i

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.s

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o: CMakeFiles/executor.dir/flags.make
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o: CMakeFiles/executor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o -MF CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o.d -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c > CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.i

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.s

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o: CMakeFiles/executor.dir/flags.make
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o: CMakeFiles/executor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o -MF CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o.d -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c > CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.i

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.s

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o: CMakeFiles/executor.dir/flags.make
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o: CMakeFiles/executor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o -MF CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o.d -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c > CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.i

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.s

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o: CMakeFiles/executor.dir/flags.make
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o: CMakeFiles/executor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o -MF CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o.d -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c > CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.i

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.s

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o: CMakeFiles/executor.dir/flags.make
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c
CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o: CMakeFiles/executor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o -MF CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o.d -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c > CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.i

CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c -o CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.s

# Object files for target executor
executor_OBJECTS = \
"CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o" \
"CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o" \
"CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o" \
"CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o" \
"CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o" \
"CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o"

# External object files for target executor
executor_EXTERNAL_OBJECTS =

/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/CHWq7_to_HWCq7.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq15.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/HWCq7_to_CHWq7_with_pad.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/executor_proxy.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/executor/src/reorder_conv2d_kernel.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/build.make
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a: CMakeFiles/executor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking C static library /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a"
	$(CMAKE_COMMAND) -P CMakeFiles/executor.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/executor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/executor.dir/build: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libexecutor.a
.PHONY : CMakeFiles/executor.dir/build

CMakeFiles/executor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/executor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/executor.dir/clean

CMakeFiles/executor.dir/depend: ../SZC_motion_1B1D_1IN.ld
CMakeFiles/executor.dir/depend: ../config.h
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles/executor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/executor.dir/depend

