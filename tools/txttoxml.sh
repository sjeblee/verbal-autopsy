#!/bin/bash

filename=$1
name="${filename%.*}.xml"


echo "<root>" > $name
cat $filename >> $name
#echo "" >> $name
echo "</root>" >> $name
