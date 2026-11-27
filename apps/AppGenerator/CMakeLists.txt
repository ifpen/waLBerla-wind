include (helper.cmake)

foreach (hardware_architecture cpu gpu)
    foreach (grid uniform staticRefinement)

        set (out_files CMakeLists.txt LBMGeneration.py input.prm)
        if (hardware_architecture STREQUAL "cpu")
            list (APPEND out_files main.cpp)
        elseif (hardware_architecture STREQUAL "gpu")
           if(NOT WALBERLA_BUILD_WITH_GPU_SUPPORT)
	        continue()
	   endif()
	   list (APPEND out_files main.cu)
        else ()
            MESSAGE(FATAL_ERROR "Invalid hardware architecture.")
        endif()

        generate_target_from_python( NAME GeneratedApp_${hardware_architecture}_${grid}
            FILE generate_app.py
            CODEGEN_ARCHITECTURE ${hardware_architecture}
            CODEGEN_GRID ${grid}
            CODEGEN_CWD ${CMAKE_CURRENT_SOURCE_DIR}
	    OUT_FILES ${out_files} 
        )

    endforeach ()
endforeach ()
