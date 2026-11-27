function( generate_target_from_python )
    set( options )
    set( oneValueArgs NAME FILE CODEGEN_ARCHITECTURE CODEGEN_GRID CODEGEN_CWD )
    set( multiValueArgs OUT_FILES)
    cmake_parse_arguments( PYGEN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    if( NOT WALBERLA_BUILD_WITH_CODEGEN )
        if( WALBERLA_LOG_SKIPPED )
            message(STATUS "Skipping ${PYGEN_NAME} since pystencils code generation is not enabled")
        endif()
        return()
    endif()

    get_filename_component(sourceFile ${PYGEN_FILE} ABSOLUTE )
    file(GLOB_RECURSE generatorInputs CONFIGURE_DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/templates/*
         ${CMAKE_CURRENT_SOURCE_DIR}/jinja_filters/*.py)

    set (sourcePath "${CMAKE_CURRENT_SOURCE_DIR}/../${PYGEN_NAME}")
    set (generationPath "${CMAKE_CURRENT_BINARY_DIR}/${PYGEN_NAME}")

    set( generatedSourceFiles ${PYGEN_OUT_FILES} )
    set( generatedWithAbsolutePath )
    foreach( filename ${generatedSourceFiles} )
        list(APPEND generatedWithAbsolutePath ${generationPath}/${filename})
    endforeach()

    set(cmakeVars "\\\{  "
            "\"WALBERLA_OPTIMIZE_FOR_LOCALHOST\": \"${WALBERLA_OPTIMIZE_FOR_LOCALHOST}\","
            "\"WALBERLA_DOUBLE_ACCURACY\": \"${WALBERLA_DOUBLE_ACCURACY}\","
            "\"CODEGEN_ARCHITECTURE\": \"${PYGEN_CODEGEN_ARCHITECTURE}\","
            "\"CODEGEN_GRID\": \"${PYGEN_CODEGEN_GRID}\","
            "\"CMAKE_CURRENT_SOURCE_DIR\": \"${CMAKE_CURRENT_SOURCE_DIR}\","
            "\"WALBERLA_BUILD_WITH_MPI\": \"${WALBERLA_BUILD_WITH_MPI}\","
            "\"WALBERLA_BUILD_WITH_CUDA\": \"${WALBERLA_BUILD_WITH_CUDA}\","
            "\"WALBERLA_BUILD_WITH_HIP\": \"${WALBERLA_BUILD_WITH_HIP}\","
            "\"WALBERLA_BUILD_WITH_OPENMP\": \"${WALBERLA_BUILD_WITH_OPENMP}\" \\\}"
    )
    string(REPLACE "\"" "\\\"" cmakeVars ${cmakeVars})   # even one more quoting level required
    string(REPLACE "\n" "" cmakeVars ${cmakeVars})  # remove newline characters

    set( WALBERLA_PYTHON_DIR ${walberla_SOURCE_DIR}/python)
    file(MAKE_DIRECTORY "${generationPath}")

    add_custom_command(OUTPUT ${generatedWithAbsolutePath}
            DEPENDS ${sourceFile} ${generatorInputs}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${WALBERLA_PYTHON_DIR}:$ENV{PYTHONPATH} ${Python_EXECUTABLE} ${sourceFile} -f ${generatedWithAbsolutePath} -c ${cmakeVars}
            WORKING_DIRECTORY "${generationPath}")

    add_library(${PYGEN_NAME} INTERFACE ${generatedWithAbsolutePath})
    # cmake might not be able to determine linker language since file extension is "hidden" in variable
    set_target_properties(${PYGEN_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(${PYGEN_NAME} INTERFACE ${generationPath})
endfunction()