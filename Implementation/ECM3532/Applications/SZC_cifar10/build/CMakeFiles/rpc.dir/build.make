# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /snap/cmake/870/bin/cmake

# The command to remove a file.
RM = /snap/cmake/870/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build

# Include any dependencies generated for this target.
include CMakeFiles/rpc.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/rpc.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/rpc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rpc.dir/flags.make

../SZC_cifar10.ld: ../config.h
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../SZC_cifar10.ld"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/linker/flash_shm_relocate_to_ram.ld.S -P -o /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/SZC_cifar10.ld -I /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10

../config.h:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "genconfig file"
	/snap/cmake/870/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/.config /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/genconfig.py --header-path /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/Kconfig

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o: CMakeFiles/rpc.dir/flags.make
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o: CMakeFiles/rpc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o -MF CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o.d -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c > CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.i

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.s

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o: CMakeFiles/rpc.dir/flags.make
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o: CMakeFiles/rpc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o -MF CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o.d -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c > CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.i

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.s

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o: CMakeFiles/rpc.dir/flags.make
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o: CMakeFiles/rpc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o -MF CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o.d -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c > CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.i

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.s

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o: CMakeFiles/rpc.dir/flags.make
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c
CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o: CMakeFiles/rpc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o -MF CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o.d -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c > CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.i

CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c -o CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.s

# Object files for target rpc
rpc_OBJECTS = \
"CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o" \
"CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o" \
"CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o" \
"CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o"

# External object files for target rpc
rpc_EXTERNAL_OBJECTS =

/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a: CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/bget.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a: CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/rpc.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a: CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/shmem.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a: CMakeFiles/rpc.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/framework/rpc/src/workQ_common.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a: CMakeFiles/rpc.dir/build.make
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a: CMakeFiles/rpc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking C static library /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a"
	$(CMAKE_COMMAND) -P CMakeFiles/rpc.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rpc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rpc.dir/build: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/librpc.a
.PHONY : CMakeFiles/rpc.dir/build

CMakeFiles/rpc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rpc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rpc.dir/clean

CMakeFiles/rpc.dir/depend: ../SZC_cifar10.ld
CMakeFiles/rpc.dir/depend: ../config.h
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles/rpc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rpc.dir/depend

