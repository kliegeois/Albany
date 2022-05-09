#!/bin/bash

SCRIPT_DIR=/projects/albany/nightlyAlbanyCDash
# Install directory holds final installed versions of the build. This is cross-mounted usually.

#Pre-processing - create ctest_nightly.cmake file from do-cmake-configure
#files.  Intel one needs to be done first, the way the scripts work right now
source $SCRIPT_DIR/convert-cmake-to-cdash.sh intel
source $SCRIPT_DIR/create-new-cdash-cmake-script.sh intel 
source $SCRIPT_DIR/convert-cmake-to-cdash.sh clang
source $SCRIPT_DIR/create-new-cdash-cmake-script.sh clang 
source $SCRIPT_DIR/convert-cmake-to-cdash.sh gcc
source $SCRIPT_DIR/create-new-cdash-cmake-script.sh gcc
cd $SCRIPT_DIR
mkdir repos
cd repos
git clone git@github.com:trilinos/Trilinos.git 
cd Trilinos
git checkout develop
cd ../
git clone git@github.com:sandialabs/Albany.git

