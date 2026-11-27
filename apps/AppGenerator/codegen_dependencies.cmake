function( turbine_core_codegen_dependencies )
    if(NOT WALBERLA_BUILD_WITH_CODEGEN)
        return()
    endif()
    file(GLOB_RECURSE codegenPackageFiles CONFIGURE_DEPENDS
         ${TurbineCore_SOURCE_DIR}/python/codegen_walberla_wind/*.py
         ${TurbineCore_SOURCE_DIR}/python/codegen_walberla_wind/templates/*)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default_codegen/waLBerlaWind_KernelInfo.h
                       APPEND DEPENDS ${codegenPackageFiles})
endfunction()
