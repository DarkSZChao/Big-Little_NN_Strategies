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
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build

# Include any dependencies generated for this target.
include CMakeFiles/freertos.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/freertos.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/freertos.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/freertos.dir/flags.make

../SZC_motion_1B6S_1IN.ld: ../config.h
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../SZC_motion_1B6S_1IN.ld"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/linker/flash_shm_relocate_to_ram.ld.S -P -o /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/SZC_motion_1B6S_1IN.ld -I /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN

../config.h:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "genconfig file"
	/snap/cmake/955/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/.config /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/genconfig.py --header-path /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/Kconfig

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.s

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.s

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.s

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.s

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.s

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.s

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.s

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o: CMakeFiles/freertos.dir/flags.make
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c
CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o: CMakeFiles/freertos.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o -MF CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o.d -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o -c /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.i"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c > CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.i

CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.s"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc --sysroot=/opt/gcc-arm-none-eabi-8-2018-q4-major/arm-none-eabi $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c -o CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.s

# Object files for target freertos
freertos_OBJECTS = \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o" \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o" \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o" \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o" \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o" \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o" \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o" \
"CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o"

# External object files for target freertos
freertos_EXTERNAL_OBJECTS =

/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/croutine.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/list.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/queue.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/tasks.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/timers.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/GCC/ARM_CM3/port.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS/Source/portable/MemMang/heap_4.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Thirdparty/FreeRTOS-Plus/Source/FreeRTOS-Plus-CLI/FreeRTOS_CLI.c.o
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/build.make
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a: CMakeFiles/freertos.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking C static library /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a"
	$(CMAKE_COMMAND) -P CMakeFiles/freertos.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freertos.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/freertos.dir/build: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libfreertos.a
.PHONY : CMakeFiles/freertos.dir/build

CMakeFiles/freertos.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/freertos.dir/cmake_clean.cmake
.PHONY : CMakeFiles/freertos.dir/clean

CMakeFiles/freertos.dir/depend: ../SZC_motion_1B6S_1IN.ld
CMakeFiles/freertos.dir/depend: ../config.h
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B6S_1IN/build/CMakeFiles/freertos.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/freertos.dir/depend

