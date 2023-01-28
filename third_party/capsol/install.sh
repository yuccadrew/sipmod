#!/bin/bash
# uncomment below if gfortran is not installed
# sudo apt-get update
# sudo apt-get install gfortran
# sudo apt-get install libblas-dev liblapack-dev

gfortran CapSol.f90 -o capsol -llapack -lblas
