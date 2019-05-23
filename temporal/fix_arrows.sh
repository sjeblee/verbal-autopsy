#!/bin/bash
# Run this in the directory containing the xml files to unescape < and >

sed -i 's/&lt;/</g' ./*
sed -i 's/&gt;/>/g' ./*
