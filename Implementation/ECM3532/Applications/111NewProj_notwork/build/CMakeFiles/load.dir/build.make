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
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build

# Utility rule file for load.

# Include any custom commands dependencies for this target.
include CMakeFiles/load.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/load.dir/progress.make

CMakeFiles/load:
	/usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/program.py --ide=gcc --soc=ecm3532 --interface=jlink --bin 111NewProj_notwork.bin --type s

load: CMakeFiles/load
load: CMakeFiles/load.dir/build.make
.PHONY : load

# Rule to build all files generated by this target.
CMakeFiles/load.dir/build: load
.PHONY : CMakeFiles/load.dir/build

CMakeFiles/load.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/load.dir/cmake_clean.cmake
.PHONY : CMakeFiles/load.dir/clean

CMakeFiles/load.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build/CMakeFiles/load.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/load.dir/depend

