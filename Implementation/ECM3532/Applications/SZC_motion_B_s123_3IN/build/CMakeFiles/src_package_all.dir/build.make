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
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build

# Utility rule file for src_package_all.

# Include any custom commands dependencies for this target.
include CMakeFiles/src_package_all.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/src_package_all.dir/progress.make

CMakeFiles/src_package_all:
	make loadconfig CONFIG=../../../Platform/ECM3532/M3/configs/src_package_all_defconfig
	make dsp_src
	make src_package

src_package_all: CMakeFiles/src_package_all
src_package_all: CMakeFiles/src_package_all.dir/build.make
.PHONY : src_package_all

# Rule to build all files generated by this target.
CMakeFiles/src_package_all.dir/build: src_package_all
.PHONY : CMakeFiles/src_package_all.dir/build

CMakeFiles/src_package_all.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/src_package_all.dir/cmake_clean.cmake
.PHONY : CMakeFiles/src_package_all.dir/clean

CMakeFiles/src_package_all.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build/CMakeFiles/src_package_all.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/src_package_all.dir/depend

