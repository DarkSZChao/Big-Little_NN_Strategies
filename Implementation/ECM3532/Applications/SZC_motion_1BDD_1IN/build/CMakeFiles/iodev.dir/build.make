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
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build

# Include any dependencies generated for this target.
include CMakeFiles/iodev.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/iodev.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/iodev.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/iodev.dir/flags.make

../SZC_motion_1BDD_1IN.ld: ../config.h
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../SZC_motion_1BDD_1IN.ld"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/linker/flash_shm_relocate_to_ram.ld.S -P -o /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/SZC_motion_1BDD_1IN.ld -I /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN

../config.h:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "genconfig file"
	/snap/cmake/955/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build/.config /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/genconfig.py --header-path /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/Kconfig

# Object files for target iodev
iodev_OBJECTS =

# External object files for target iodev
iodev_EXTERNAL_OBJECTS =

/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libiodev.a: CMakeFiles/iodev.dir/build.make
/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libiodev.a: CMakeFiles/iodev.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C static library /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libiodev.a"
	$(CMAKE_COMMAND) -P CMakeFiles/iodev.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/iodev.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/iodev.dir/build: /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/pre_built/libiodev.a
.PHONY : CMakeFiles/iodev.dir/build

CMakeFiles/iodev.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/iodev.dir/cmake_clean.cmake
.PHONY : CMakeFiles/iodev.dir/clean

CMakeFiles/iodev.dir/depend: ../SZC_motion_1BDD_1IN.ld
CMakeFiles/iodev.dir/depend: ../config.h
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1BDD_1IN/build/CMakeFiles/iodev.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/iodev.dir/depend

