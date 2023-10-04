#cmake_minimum_required (VERSION 2.8)
set (CTEST_DO_SUBMIT ON)
set (CTEST_TEST_TYPE Nightly)
SET (CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")
set (CTEST_CONFIGURATION "Release") 
message("CTEST_BUILD_OPTION = " ${CTEST_BUILD_OPTION})

execute_process(COMMAND bash delete_txt_files.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set (TRILINSTALLDIR "/home/projects/albany/nightlyCDashTrilinosBlake/build-gcc/TrilinosReleaseInstallGcc")
execute_process(COMMAND grep "Trilinos_C_COMPILER " ${TRILINSTALLDIR}/lib64/cmake/Trilinos/TrilinosConfig.cmake
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE MPICC_RESULT
		OUTPUT_FILE "mpicc.txt")
execute_process(COMMAND bash get_mpicc.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE GET_MPICC_RESULT)
execute_process(COMMAND cat mpicc.txt 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE GET_MPICC_RESULT
		OUTPUT_VARIABLE MPICC
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT mpicc = " ${MPICC}) 
execute_process(COMMAND ${MPICC} -dumpversion 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE COMPILER_VERSION_RESULT
		OUTPUT_VARIABLE COMPILER_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT compiler version = " ${COMPILER_VERSION})
execute_process(COMMAND ${MPICC} --version 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE COMPILER_RESULT
		OUTPUT_FILE "compiler.txt")
execute_process(COMMAND bash process_compiler.sh 
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		RESULT_VARIABLE CHANGE_COMPILER_RESULT
		OUTPUT_VARIABLE COMPILER
		OUTPUT_STRIP_TRAILING_WHITESPACE)
#message("IKT compiler = " ${COMPILER})


find_program(UNAME NAMES uname)
macro(getuname name flag)
  exec_program("${UNAME}" ARGS "${flag}" OUTPUT_VARIABLE "${name}")
endmacro(getuname)

getuname(osname -s)
getuname(osrel  -r)
getuname(cpu    -m)

#message("IKT osname = " ${osname}) 
#message("IKT osrel = " ${osrel}) 
#message("IKT cpu = " ${cpu}) 

set (CTEST_BUILD_NAME "Albany-${osname}-${osrel}-${COMPILER}-${COMPILER_VERSION}-${CTEST_CONFIGURATION}-${CTEST_BUILD_OPTION}")

# What to build and test
set (CLEAN_BUILD FALSE)
set (DOWNLOAD_ALBANY FALSE) 
if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "sfad6")
    set (BUILD_ALBANY_GCC_SFAD6 TRUE) 
    set (BUILD_ALBANY_GCC_SFAD12 FALSE) 
    set (BUILD_ALBANY_GCC_SFAD24 FALSE) 
    set (SFAD_SIZE 6)  
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "sfad12")
    set (BUILD_ALBANY_GCC_SFAD6 FALSE) 
    set (BUILD_ALBANY_GCC_SFAD12 TRUE) 
    set (BUILD_ALBANY_GCC_SFAD24 FALSE) 
    set (SFAD_SIZE 12)  
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "sfad24")
    set (BUILD_ALBANY_GCC_SFAD6 FALSE) 
    set (BUILD_ALBANY_GCC_SFAD12 FALSE) 
    set (BUILD_ALBANY_GCC_SFAD24 TRUE) 
    set (SFAD_SIZE 24)  
  ENDIF()
ENDIF()



# Begin User inputs:
set (CTEST_SITE "blake.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos-gcc)
set (CTEST_BINARY_NAME build-gcc)


set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_BUILD_FLAGS "-j48")

find_program (CTEST_GIT_COMMAND NAMES git)

set (Albany_REPOSITORY_LOCATION git@github.com:sandialabs/Albany.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (MPI_PATH $ENV{MPI_ROOT})  
set (MKL_PATH $ENV{MKL_ROOT})  
set (SUPERLU_PATH $ENV{SUPERLU_ROOT})  
set (BOOST_PATH $ENV{BOOST_ROOT}) 
set (NETCDF_PATH $ENV{NETCDF_ROOT}) 
set (HDF5_PATH $ENV{HDF5_ROOT})
set (ZLIB_PATH $ENV{ZLIB_ROOT})  

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()


if (DOWNLOAD_ALBANY)

  set (CTEST_CHECKOUT_COMMAND)
  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  #
  # Get Albany
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Albany")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Albany_REPOSITORY_LOCATION} -b master ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Albany repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  # Pull the repo
  execute_process (COMMAND "${CTEST_GIT_COMMAND}" pull
      WORKING_DIRECTORY ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
  message(STATUS "Output of Albany pull: ${_out}")
  message(STATUS "Text sent to standard error stream: ${_err}")
  message(STATUS "command result status: ${HAD_ERROR}")
  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot pull Albany!")
  endif ()

endif ()


ctest_start(${CTEST_TEST_TYPE})


# Configure the Albany build 
#

set (CONFIGURE_OPTIONS
    CDASH-ALBANY-FILE.TXT
)

IF (BUILD_ALBANY_GCC_SFAD6) 
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad6")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad6)
  endif ()
  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad6"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()
IF (BUILD_ALBANY_GCC_SFAD12) 
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad12")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad12)
  endif ()
  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad12"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()
IF (BUILD_ALBANY_GCC_SFAD24) 
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad24")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad24)
  endif ()
  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad24"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )
ENDIF()


if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Configure
    RETURN_VALUE  S_HAD_ERROR
    )

  if (S_HAD_ERROR)
    message ("Cannot submit Albany configure results!")
  endif ()
endif ()

if (HAD_ERROR)
  message ("Cannot configure Albany build!")
endif ()

#
# Build the rest of Albany and install everything
#

set (CTEST_BUILD_TARGET all)
#set (CTEST_BUILD_TARGET install)

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

IF (BUILD_ALBANY_GCC_SFAD6) 
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad6"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )
ENDIF()
IF (BUILD_ALBANY_GCC_SFAD12) 
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad12"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )
ENDIF()
IF (BUILD_ALBANY_GCC_SFAD24) 
  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad24"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )
ENDIF()


if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Build
    RETURN_VALUE  S_HAD_ERROR
    )

  if (S_HAD_ERROR)
    message ("Cannot submit Albany build results!")
  endif ()

endif ()

if (HAD_ERROR)
  message ("Cannot build Albany!")
endif ()

if (BUILD_LIBS_NUM_ERRORS GREATER 0)
  message ("Encountered build errors in Albany build. Exiting!")
endif ()

#
# Run Albany tests
#
#  Over-write default limit for output posted to CDash site
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

set (CTEST_TEST_TIMEOUT 600)

IF (BUILD_ALBANY_GCC_SFAD6) 
  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad6"
    RETURN_VALUE  HAD_ERROR
    )
ENDIF()
IF (BUILD_ALBANY_GCC_SFAD12) 
  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad12"
    RETURN_VALUE  HAD_ERROR
    )
ENDIF()
IF (BUILD_ALBANY_GCC_SFAD24) 
  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/AlbBuildGccSFad24"
    RETURN_VALUE  HAD_ERROR
    )
ENDIF()

if (CTEST_DO_SUBMIT)
  ctest_submit (PARTS Test
    RETURN_VALUE  S_HAD_ERROR
    )

  if (S_HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Albany test results!")
  endif ()
endif ()

