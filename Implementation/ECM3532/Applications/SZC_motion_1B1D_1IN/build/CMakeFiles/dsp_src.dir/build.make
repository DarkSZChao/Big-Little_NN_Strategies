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

# Utility rule file for dsp_src.

# Include any custom commands dependencies for this target.
include CMakeFiles/dsp_src.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dsp_src.dir/progress.make

CMakeFiles/dsp_src: CMakeFiles/dsp_src-complete

CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-install
CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-mkdir
CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-download
CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-update
CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-patch
CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-configure
CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-build
CMakeFiles/dsp_src-complete: dsp_src-prefix/src/dsp_src-stamp/dsp_src-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'dsp_src'"
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles
	/snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles/dsp_src-complete
	/snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-done

dsp_src-prefix/src/dsp_src-stamp/dsp_src-build: dsp_src-prefix/src/dsp_src-stamp/dsp_src-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Performing build step for 'dsp_src'"
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && cmake .. && make loadconfig CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test/../../configs/dsp_src_package_defconfig && make && make src_package
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-build

dsp_src-prefix/src/dsp_src-stamp/dsp_src-configure: dsp_src-prefix/tmp/dsp_src-cfgcmd.txt
dsp_src-prefix/src/dsp_src-stamp/dsp_src-configure: dsp_src-prefix/src/dsp_src-stamp/dsp_src-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Performing configure step for 'dsp_src'"
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/955/bin/cmake "-GUnix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-configure

dsp_src-prefix/src/dsp_src-stamp/dsp_src-download: dsp_src-prefix/src/dsp_src-stamp/dsp_src-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "No download step for 'dsp_src'"
	/snap/cmake/955/bin/cmake -E echo_append
	/snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-download

dsp_src-prefix/src/dsp_src-stamp/dsp_src-install: dsp_src-prefix/src/dsp_src-stamp/dsp_src-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing install step for 'dsp_src'"
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/955/bin/cmake -E copy /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test/build/DSP.tar.gz /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-install

dsp_src-prefix/src/dsp_src-stamp/dsp_src-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'dsp_src'"
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test/build
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/tmp
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src
	/snap/cmake/955/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp
	/snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-mkdir

dsp_src-prefix/src/dsp_src-stamp/dsp_src-patch: dsp_src-prefix/src/dsp_src-stamp/dsp_src-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'dsp_src'"
	/snap/cmake/955/bin/cmake -E echo_append
	/snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-patch

dsp_src-prefix/src/dsp_src-stamp/dsp_src-update: dsp_src-prefix/src/dsp_src-stamp/dsp_src-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No update step for 'dsp_src'"
	/snap/cmake/955/bin/cmake -E echo_append
	/snap/cmake/955/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/dsp_src-prefix/src/dsp_src-stamp/dsp_src-update

dsp_src: CMakeFiles/dsp_src
dsp_src: CMakeFiles/dsp_src-complete
dsp_src: dsp_src-prefix/src/dsp_src-stamp/dsp_src-build
dsp_src: dsp_src-prefix/src/dsp_src-stamp/dsp_src-configure
dsp_src: dsp_src-prefix/src/dsp_src-stamp/dsp_src-download
dsp_src: dsp_src-prefix/src/dsp_src-stamp/dsp_src-install
dsp_src: dsp_src-prefix/src/dsp_src-stamp/dsp_src-mkdir
dsp_src: dsp_src-prefix/src/dsp_src-stamp/dsp_src-patch
dsp_src: dsp_src-prefix/src/dsp_src-stamp/dsp_src-update
dsp_src: CMakeFiles/dsp_src.dir/build.make
.PHONY : dsp_src

# Rule to build all files generated by this target.
CMakeFiles/dsp_src.dir/build: dsp_src
.PHONY : CMakeFiles/dsp_src.dir/build

CMakeFiles/dsp_src.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dsp_src.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dsp_src.dir/clean

CMakeFiles/dsp_src.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_1B1D_1IN/build/CMakeFiles/dsp_src.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dsp_src.dir/depend

