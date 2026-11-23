
include_directories( ${TurbineCore_SOURCE_DIR}/src )
include_directories( ${TurbineCore_BINARY_DIR}/src ) # for generated headers

add_subdirectory(component)
add_subdirectory(domain)
add_subdirectory(walberla_helper)
