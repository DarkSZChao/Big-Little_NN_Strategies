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
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test/build

# Utility rule file for dsp_clean.

# Include any custom commands dependencies for this target.
include CMakeFiles/dsp_clean.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dsp_clean.dir/progress.make

CMakeFiles/dsp_clean:
	rm -rf /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test/build/dsp_fw-prefix/
	rm -rf /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test/build/dsp_src-prefix/

dsp_clean: CMakeFiles/dsp_clean
dsp_clean: CMakeFiles/dsp_clean.dir/build.make
.PHONY : dsp_clean

# Rule to build all files generated by this target.
CMakeFiles/dsp_clean.dir/build: dsp_clean
.PHONY : CMakeFiles/dsp_clean.dir/build

CMakeFiles/dsp_clean.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dsp_clean.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dsp_clean.dir/clean

CMakeFiles/dsp_clean.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10_test/build/CMakeFiles/dsp_clean.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dsp_clean.dir/depend
