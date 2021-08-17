# Name the test with the directory name
get_filename_component(testNameRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)

####################################
###          Main tests          ###
####################################

set(testName ${testNameRoot})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/heat_2d.yaml
${CMAKE_CURRENT_BINARY_DIR}/heat_2d.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e
${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.0
${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.1
${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.2
${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.3
${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.3 COPYONLY)



add_test(${testName}_Heat_2D ${Albany.exe} heat_2d.yaml)
set_tests_properties(${testName}_Heat_2D PROPERTIES LABELS "Basic;Tpetra;Forward")
