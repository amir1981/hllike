This directory contains HLlike.c : This program works on Top of HTK and compute
log-likelihood for each model.
To install:
1- First install HTK
2- set HTK_PATH to  directory contains HTK (e.g. export HTK_PATH=../htk)
3- run sh prepare.sh
4- Go to src directory, open Makefile (isip_e Makefile), broswe the Makefile until you find  "PROGS  =  ". Now Remove everything in right of the equation and insert  HLlike HLlike_space  instead.
example:
PROGS =  HLlike  HLlike_space 

5- inside src type: make
	            make_install

This will install HLLike  and HLLike_space inside $HTK_PATH/bin
