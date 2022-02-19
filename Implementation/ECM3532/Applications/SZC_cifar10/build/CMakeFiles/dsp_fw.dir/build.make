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

# Utility rule file for dsp_fw.

# Include any custom commands dependencies for this target.
include CMakeFiles/dsp_fw.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dsp_fw.dir/progress.make

CMakeFiles/dsp_fw: CMakeFiles/dsp_fw-complete

CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-install
CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-mkdir
CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-download
CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-update
CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-patch
CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-configure
CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-build
CMakeFiles/dsp_fw-complete: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'dsp_fw'"
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles
	/snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles/dsp_fw-complete
	/snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-done

dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-build: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Performing build step for 'dsp_fw'"
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/870/bin/cmake -E copy /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test/CMakeLists.txt
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && cmake .. && make loadconfig CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/configs/executor_test_dsp_defconfig && make
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-build

dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-configure: dsp_fw-prefix/tmp/dsp_fw-cfgcmd.txt
dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-configure: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Performing configure step for 'dsp_fw'"
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/870/bin/cmake "-GUnix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-configure

dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-download: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "No download step for 'dsp_fw'"
	/snap/cmake/870/bin/cmake -E echo_append
	/snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-download

dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-install: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing install step for 'dsp_fw'"
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/870/bin/cmake -E copy /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test/build/dsp_fw.bin /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/util/dsp_fw/dsp_fw_executor.bin
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/DSP/app/executor_test/build && /snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-install

dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Creating directories for 'dsp_fw'"
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../DSP/app/executor_test/build
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/tmp
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src
	/snap/cmake/870/bin/cmake -E make_directory /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp
	/snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-mkdir

dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-patch: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-update
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No patch step for 'dsp_fw'"
	/snap/cmake/870/bin/cmake -E echo_append
	/snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-patch

dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-update: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No update step for 'dsp_fw'"
	/snap/cmake/870/bin/cmake -E echo_append
	/snap/cmake/870/bin/cmake -E touch /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-update

dsp_fw: CMakeFiles/dsp_fw
dsp_fw: CMakeFiles/dsp_fw-complete
dsp_fw: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-build
dsp_fw: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-configure
dsp_fw: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-download
dsp_fw: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-install
dsp_fw: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-mkdir
dsp_fw: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-patch
dsp_fw: dsp_fw-prefix/src/dsp_fw-stamp/dsp_fw-update
dsp_fw: CMakeFiles/dsp_fw.dir/build.make
.PHONY : dsp_fw

# Rule to build all files generated by this target.
CMakeFiles/dsp_fw.dir/build: dsp_fw
.PHONY : CMakeFiles/dsp_fw.dir/build

CMakeFiles/dsp_fw.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dsp_fw.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dsp_fw.dir/clean

CMakeFiles/dsp_fw.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_cifar10/build/CMakeFiles/dsp_fw.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dsp_fw.dir/depend

