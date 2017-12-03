#!/bin/bash

file=$1
outfile=$2

# remove blank lines | remove extra spaces| remove leading and trailing spaces  | fix utf-8 chars
sed '/^\s*$/d' $file | sed -e 's/  */ /g' | sed -e 's/^ //g' | sed -e 's/ $//g' | sed -e 's/&amp;/and/g' | sed -e 's/&#13;/ /g' | sed -e "s/&#8217;/\'/g" | sed -e 's/&#8221;/"/g' | sed -e 's/&#8220;/"/g' | sed -e 's/&#65533;//g' | sed -e 's/&#1757;//g'| sed -e "s/&#1770;/\'/g" > $outfile
