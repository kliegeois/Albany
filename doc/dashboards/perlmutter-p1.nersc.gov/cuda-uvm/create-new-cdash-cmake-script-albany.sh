#!/bin/bash

sed -e "/CDASH-ALBANY-FILE.TXT/r /pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm/cdash-albany-frag.txt" -e "/CDASH-ALBANY-FILE.TXT/d" "/pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm/ctest_nightly_albany_tmp.cmake" >& /pscratch/sd/m/mcarlson/biweeklyCDashPerlmutter-cuda-uvm/ctest_nightly_albany.cmake
