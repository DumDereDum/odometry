cmake_minimum_required(VERSION 2.8)

project(odometry)

set(MAIN
	odometry/main.cpp
	)
set(INCLUDE
	odometry/include/tmp.hpp
	odometry/include/odometry.hpp
)
set(ODOMETRY_SETTINGS
	odometry/src/odometry_settings.hpp
	odometry/src/odometry_settings_impl.cpp
	)
set(ODOMETRY_FRAME
	odometry/src/odometry_frame.hpp
	odometry/src/odometry_frame_impl.cpp
	)
set(ODOMETRY_IMPL
	odometry/src/odometry.cpp
	odometry/src/odometry_icp.hpp
	odometry/src/odometry_icp.cpp
	odometry/src/odometry_rgb.hpp
	odometry/src/odometry_rgb.cpp
	odometry/src/odometry_rgbd.hpp
	odometry/src/odometry_rgbd.cpp
	odometry/src/odometry_functions.hpp
	odometry/src/odometry_functions.cpp
	)
set(DEPTH_TO_3D
	odometry/include/depth_to_3d.hpp
	odometry/include/depth_to_3d.cpp
)
set(NORMALS
	odometry/src/normal.hpp
	odometry/src/normal.cpp
)

add_executable(main ${MAIN} ${INCLUDE} ${ODOMETRY_SETTINGS} ${ODOMETRY_FRAME} ${ODOMETRY_IMPL} ${DEPTH_TO_3D} ${NORMALS})
