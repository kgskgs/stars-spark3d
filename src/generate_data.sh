#!/bin/bash


SEED=0
OUTDIR=.

while getopts s:o: option
do
    case "${option}"
    in
        s) SEED=${OPTARG};;
        o) OUTDIR=${OPTARG};;
    esac
done
shift $(expr $OPTIND - 1 )

for N in $@
do
    mkplummer - $N seed=$SEED |
    snapprint - x,y,z,vx,vy,vz,m csv=t comment=t |
    { sed -u 's/\s\+/,/g;s/#/id/;q'; nl -s, -w1; } \
    > "${OUTDIR}/plummer_${N}_s${SEED}.csv"
done