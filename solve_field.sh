 #!/bin/bash

#solve field
solve-field --scale-low 0.1 --scale-high 1 -O -o $1_wcs -3 $2 -4 $3 -5 1 --skip-solved $4 --temp-dir $5
#1 = base file name, no path needed, no .fit needed
#2 = center RA
#3 = center Dec
#4 = orignal file path
#5 = temporary files directory
exit
