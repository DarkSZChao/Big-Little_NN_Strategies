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
CMAKE_SOURCE_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build

# Utility rule file for ldconf.

# Include any custom commands dependencies for this target.
include CMakeFiles/ldconf.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ldconf.dir/progress.make

CMakeFiles/ldconf: ../SZC_motion_energy_measurement.ld

../SZC_motion_energy_measurement.ld: ../config.h
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../SZC_motion_energy_measurement.ld"
	/opt/gcc-arm-none-eabi-8-2018-q4-major/bin/arm-none-eabi-gcc -E /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/scripts/linker/flash_shm_relocate_to_ram.ld.S -P -o /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/SZC_motion_energy_measurement.ld -I /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement

../config.h:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "genconfig file"
	/snap/cmake/955/bin/cmake -E env KCONFIG_CONFIG=/home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/.config /usr/bin/python3.8 /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/../../../Thirdparty/Kconfiglib/genconfig.py --header-path /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/config.h /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Platform/ECM3532/M3/Kconfig

ldconf: ../SZC_motion_energy_measurement.ld
ldconf: CMakeFiles/ldconf
ldconf: ../config.h
ldconf: CMakeFiles/ldconf.dir/build.make
.PHONY : ldconf

# Rule to build all files generated by this target.
CMakeFiles/ldconf.dir/build: ldconf
.PHONY : CMakeFiles/ldconf.dir/build

CMakeFiles/ldconf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ldconf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ldconf.dir/clean

CMakeFiles/ldconf.dir/depend:
	cd /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build /home/szc/Desktop/TensaiFlow_rc_alpha2-0.2/Applications/SZC_motion_energy_measurement/build/CMakeFiles/ldconf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ldconf.dir/depend

