#1/bin/bash

filename=$1
name="${filename%.*}"
sed '$d' < $filename | sed "1d" > $name.txt 
