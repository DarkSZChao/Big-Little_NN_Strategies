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

# Utility rule file for flash_app.

# Include any custom commands dependencies for this target.
include CMakeFiles/flash_app.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/flash_app.dir/progress.make

CMakeFiles/flash_app:
	/usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/program.py --ide=gcc --soc=ecm3532 --interface=jlink --bin SZC_cifar10.bin --type a

flash_app: CMakeFiles/flash_app
flash_app: CMakeFiles/flash_app.dir/build.make
.PHONY : flash_app

# Rule to build all files generated by this target.
CMakeFiles/flash_app.dir/build: flash_app
.PHONY : CMakeFiles/flash_app.dir/build

CMakeFiles/flash_app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/flash_app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/flash_app.dir/clean

CMakeFiles/flash_app.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles/flash_app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/flash_app.dir/depend

