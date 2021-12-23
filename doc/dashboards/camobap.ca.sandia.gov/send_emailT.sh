#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /mnt/encrypted_sdc1/nightlyCDash/nightly_logT.txt -c`
TTTT=`grep "(Not Run)" /mnt/encrypted_sdc1/nightlyCDash/nightly_logT.txt -c`
TTTTT=`grep "(Timeout)" /mnt/encrypted_sdc1/nightlyCDash/nightly_logT.txt -c`
TT=`grep "...   Passed" /mnt/encrypted_sdc1/nightlyCDash/nightly_logT.txt -c`

#/bin/mail -s "Albany ($ALBANY_BRANCH): $TTT" "albany-regression@software.sandia.gov" < $ALBOUTDIR/albany_runtests.out
/bin/mail -s "IKTAlbanyNoEpetra, camobap.ca.sandia.gov: $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, daibane@sandia.gov, tjfulle@sandia.gov, lbertag@sandia.gov, mperego@sandia.gov" < /mnt/encrypted_sdc1/nightlyCDash/resultsT
