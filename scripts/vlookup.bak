#!/bin/bash

### Seen it in stackexchange.com '('https://unix.stackexchange.com/questions/88550/vlookup-function-in-unix')' and modified by RFdS
## Explained:
# This will read the first file in the command $awk -f vlookup.awk <file1> <file2> into an array - a - and then will match the 5th column of file 2. If true, will print the line to STDOUT

FNR==NR{
  a[$1]=$2
  next
}
{ if ($5 in a) {print $0} }