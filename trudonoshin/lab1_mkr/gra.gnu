set size square
set terminal postscript eps enhanced color font 'Helvetica,10'
set output 'plot.eps'
set cbrange [0:10]

set pm3d map
set pm3d flush begin ftriangles scansforwar interpolate 5,5
splot 'data.txt' using 1:2:3 with pm3d title 'var'
