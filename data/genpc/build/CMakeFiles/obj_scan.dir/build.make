# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wuhang/GrowPC/data/genpc/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wuhang/GrowPC/data/genpc/build

# Include any dependencies generated for this target.
include CMakeFiles/obj_scan.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/obj_scan.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/obj_scan.dir/flags.make

CMakeFiles/obj_scan.dir/obj_scan.cpp.o: CMakeFiles/obj_scan.dir/flags.make
CMakeFiles/obj_scan.dir/obj_scan.cpp.o: /home/wuhang/GrowPC/data/genpc/src/obj_scan.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wuhang/GrowPC/data/genpc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/obj_scan.dir/obj_scan.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/obj_scan.dir/obj_scan.cpp.o -c /home/wuhang/GrowPC/data/genpc/src/obj_scan.cpp

CMakeFiles/obj_scan.dir/obj_scan.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obj_scan.dir/obj_scan.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wuhang/GrowPC/data/genpc/src/obj_scan.cpp > CMakeFiles/obj_scan.dir/obj_scan.cpp.i

CMakeFiles/obj_scan.dir/obj_scan.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obj_scan.dir/obj_scan.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wuhang/GrowPC/data/genpc/src/obj_scan.cpp -o CMakeFiles/obj_scan.dir/obj_scan.cpp.s

CMakeFiles/obj_scan.dir/obj_scan.cpp.o.requires:

.PHONY : CMakeFiles/obj_scan.dir/obj_scan.cpp.o.requires

CMakeFiles/obj_scan.dir/obj_scan.cpp.o.provides: CMakeFiles/obj_scan.dir/obj_scan.cpp.o.requires
	$(MAKE) -f CMakeFiles/obj_scan.dir/build.make CMakeFiles/obj_scan.dir/obj_scan.cpp.o.provides.build
.PHONY : CMakeFiles/obj_scan.dir/obj_scan.cpp.o.provides

CMakeFiles/obj_scan.dir/obj_scan.cpp.o.provides.build: CMakeFiles/obj_scan.dir/obj_scan.cpp.o


# Object files for target obj_scan
obj_scan_OBJECTS = \
"CMakeFiles/obj_scan.dir/obj_scan.cpp.o"

# External object files for target obj_scan
obj_scan_EXTERNAL_OBJECTS =

libobj_scan.so: CMakeFiles/obj_scan.dir/obj_scan.cpp.o
libobj_scan.so: CMakeFiles/obj_scan.dir/build.make
libobj_scan.so: CMakeFiles/obj_scan.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wuhang/GrowPC/data/genpc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libobj_scan.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/obj_scan.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/obj_scan.dir/build: libobj_scan.so

.PHONY : CMakeFiles/obj_scan.dir/build

CMakeFiles/obj_scan.dir/requires: CMakeFiles/obj_scan.dir/obj_scan.cpp.o.requires

.PHONY : CMakeFiles/obj_scan.dir/requires

CMakeFiles/obj_scan.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/obj_scan.dir/cmake_clean.cmake
.PHONY : CMakeFiles/obj_scan.dir/clean

CMakeFiles/obj_scan.dir/depend:
	cd /home/wuhang/GrowPC/data/genpc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wuhang/GrowPC/data/genpc/src /home/wuhang/GrowPC/data/genpc/src /home/wuhang/GrowPC/data/genpc/build /home/wuhang/GrowPC/data/genpc/build /home/wuhang/GrowPC/data/genpc/build/CMakeFiles/obj_scan.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/obj_scan.dir/depend
