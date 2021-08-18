cmake_minimum_required(VERSION 2.8)

project(odometry)

set(MAIN
	src/main.cpp
	)
set(ODOMETRY
	src/odometry.hpp
	src/odometry.cpp
	)
set(ODOMETRY_FRAME
	src/odometry_func/odometry_frame.hpp
	src/odometry_func/odometry_frame.cpp
	)
set(ODOMETRY_IMPL
	src/odometry_func/odometry_impl.hpp
	src/odometry_func/odometry_impl.cpp
	src/odometry_func/odometry_icp.hpp
	src/odometry_func/odometry_icp.cpp
	src/odometry_func/odometry_rgb.hpp
	src/odometry_func/odometry_rgb.cpp
	src/odometry_func/odometry_rgbd.hpp
	src/odometry_func/odometry_rgbd.cpp
	src/odometry_func/odometry_functions.hpp
	src/odometry_func/odometry_functions.cpp
	)
	
add_executable(main ${MAIN} ${ODOMETRY} ${ODOMETRY_FRAME} ${ODOMETRY_IMPL})
