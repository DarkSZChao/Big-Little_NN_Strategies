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

# Utility rule file for autoconf.

# Include any custom commands dependencies for this target.
include CMakeFiles/autoconf.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/autoconf.dir/progress.make

CMakeFiles/autoconf: ../config.h

../config.h:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "genconfig file"
	/snap/cmake/870/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build/.config /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/genconfig.py --header-path /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/Kconfig

autoconf: CMakeFiles/autoconf
autoconf: ../config.h
autoconf: CMakeFiles/autoconf.dir/build.make
.PHONY : autoconf

# Rule to build all files generated by this target.
CMakeFiles/autoconf.dir/build: autoconf
.PHONY : CMakeFiles/autoconf.dir/build

CMakeFiles/autoconf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/autoconf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/autoconf.dir/clean

CMakeFiles/autoconf.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/111NewProj_notwork/build/CMakeFiles/autoconf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/autoconf.dir/depend

