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

# Utility rule file for loadconfig.

# Include any custom commands dependencies for this target.
include CMakeFiles/loadconfig.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/loadconfig.dir/progress.make

CMakeFiles/loadconfig:
	/snap/cmake/955/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build/.config srctree=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3 /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/defconfig.py ${CONFIG}
	cmake .. >/dev/null
	/snap/cmake/955/bin/cmake -E remove -f /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/config.h
	make ses
	make dsp_update

loadconfig: CMakeFiles/loadconfig
loadconfig: CMakeFiles/loadconfig.dir/build.make
.PHONY : loadconfig

# Rule to build all files generated by this target.
CMakeFiles/loadconfig.dir/build: loadconfig
.PHONY : CMakeFiles/loadconfig.dir/build

CMakeFiles/loadconfig.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/loadconfig.dir/cmake_clean.cmake
.PHONY : CMakeFiles/loadconfig.dir/clean

CMakeFiles/loadconfig.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_B_s123/build/CMakeFiles/loadconfig.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/loadconfig.dir/depend
