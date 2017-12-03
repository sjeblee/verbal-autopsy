#!/usr/bin/python
# -*- coding: utf-8 -*-
# Convert the csv of the translated narratives to an xml tree

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')

from lxml import etree
import argparse
import calendar
import csv
import re
import subprocess

import data_util

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./convert_csv_to_xml.py --in [file.xml] --out [outfile.txt]"
        exit()

    # Create the xml root
    adult_root = etree.Element("root")
    child_root = etree.Element("root")
    neonate_root = etree.Element("root")
    adult_tag = "Adult_Anonymous"
    child_tag = "Child_Anonymous"
    neonate_tag = "Neonate_Anonymous"
    id_tag = "MG_ID"
    narr_tag = "narrative"
    keywords1_tag = "CODINGKEYWORDS1"
    keywords2_tag = "CODINGKEYWORDS2"
    icd1_tag = "CODINGICDCODE1"
    icd2_tag = "CODINGICDCODE2"
    rec1_icd_tag = "RECON_ICDCODE_1"
    rec2_icd_tag = "RECON_ICDCODE_2"
    rec1_keywords_tag = "RECON_KEYWORDS_1"
    rec2_keywords_tag = "RECON_KEYWORDS_2"
    icd_tag = "Final_code"
    age_tag = "DeathAge"
    ageunit_tag = "ageunit"
    deathdate_tag = "DeathDate"
    gender_tag = "DeceasedSex"

    # Parse CSV file
    id_name = "id"
    age_name = "age_value"
    ageunit_name = "age_unit"
    gender_name = "sex"
    icd_name = "final_icd"
    narr_name = "translated_eva_narrative"

    # Anand
    #if "anand" in args.infile:
        #print "Anand"
    keywords1_name = "physician_one_coding_comments"
    keywords2_name = "physician_two_coding_comments"
    adj_icd_name = "adjudication_icd"
    rec1_icd_name = "physician_one_reconciliation_icd"
    rec2_icd_name = "physician_two_reconciliation_icd"
    rec1_keywords_name = "physician_one_reconciliation_comments"
    rec2_keywords_name = "physician_two_reconciliation_comments"
    phys1_icd_name = "physician_one_coding_icd"
    phys2_icd_name = "physician_two_coding_icd"
    death_day_name = "dod_day"
    death_month_name = "dod_month"
    death_year_name = "dod_year"

    if "amravati" in args.infile:
        print "Amravati"
        death_day_name = "dod"
        death_month_name = "monthod"
        death_year_name = "yod"
        keywords1_name = "physician1CodingKeywords"
        keywords2_name = "physician2CodingKeywords"
        adj_icd_name = "adjudicatorICD"
        rec1_icd_name = "physician1ReconciliationICD"
        rec2_icd_name = "physician2ReconciliationICD"
        rec1_keywords_name = "physician1ReconciliationKeywords"
        rec2_keywords_name = "physician2ReconciliationKeywords"
        phys1_icd_name = "physician1CodingICD"
        phys2_icd_name = "physician2CodingICD"

    elif "punjab" in args.infile:
        # death dates the same as Anand
        keywords1_name = "Phy1_CodingComments"
        keywords2_name = "Phy2_CodingComments"
        adj_icd_name = "Adjudication_ICD"
        rec1_icd_name = "Phy1_ReconciliationICD"
        rec2_icd_name = "Phy2_ReconciliationICD"
        rec1_keywords_name = "Phy1_ReconciliationComments"
        rec2_keywords_name = "Phy2_ReconciliationComments"
        phys1_icd_name = "Physician_1_CodingICD"
        phys2_icd_name = "Physician_2_CodingICD"

    dropped = 0
    with open(args.infile, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #print "row: " + row
            root = adult_root
            tag = adult_tag
            age_str = row[age_name]
            if not age_str == "NULL":
                age = int(age_str)
                ageunit = row[ageunit_name]
                print "age: " + str(age) + " " + ageunit
                if (age < 1 and ageunit == "Months") or (ageunit == "Days"):
                    print "neonate"
                    tag = neonate_tag
                    root = neonate_root
                elif (age < 15 and ageunit == "Years") or (ageunit == "Months"):
                    print "child"
                    tag = child_tag
                    root = child_root
            child = etree.SubElement(root, tag)
            id_node = etree.SubElement(child, id_tag)
            id_node.text = row[id_name]
            age_node = etree.SubElement(child, age_tag)
            age_node.text = str(age)
            ageunit_node = etree.SubElement(child, ageunit_tag)
            ageunit_node.text = ageunit
            gender = row[gender_name]
            #print "gender: " + gender
            if gender == "Female":
                gender = 2
            elif gender == "Male":
                gender = 1
            gender_node = etree.SubElement(child, gender_tag)
            gender_node.text = str(gender)
            death_node = etree.SubElement(child, deathdate_tag)
            deathdate = ""
            if death_year_name in row:
                deathdate = row[death_year_name]
            else:
                deathdate = "????"
            if death_month_name in row:
                month = row[death_month_name]
                if not month.isdigit():
                    month = month_to_num(month)
                deathdate = deathdate + "-" + month
            else:
                deathdate = deathdate + "-?"
            if death_day_name in row:
                deathdate = deathdate + "-" + row[death_day_name]
            else:
                deathdate = deathdate + "-?"
            death_node.text = deathdate

            # Get keywords and fix the commas
            keywords1 = row[keywords1_name]
            keywords2 = row[keywords2_name]
            keywords1_node = etree.SubElement(child, keywords1_tag)
            keywords1_node.text = fix_keywords(keywords1)
            keywords2_node = etree.SubElement(child, keywords2_tag)
            keywords2_node.text = fix_keywords(keywords2)

            # Get narrative
            narr_node = etree.SubElement(child, narr_tag)
            narr_node.text = fix_narr(row[narr_name])
            #print "narr: " + narr_node.text

            # Get ICD codes and reconciliation keywords
            icd = ""
            icd1 = row[phys1_icd_name]
            icd2 = row[phys2_icd_name]
            icd1_node = etree.SubElement(child, icd1_tag)
            icd2_node = etree.SubElement(child, icd2_tag)
            icd1_node.text = icd1
            icd2_node.text = icd2
            if icd1.upper() != icd2.upper():
                rec_icd1 = row[rec1_icd_name]
                rec_icd2 = row[rec2_icd_name]
                if rec_icd1 == "" and rec_icd2 == "" and icd1[0] == icd2[0]:
                    icd = icd1
                else:
                    icd1_rec_node = etree.SubElement(child, rec1_icd_tag)
                    icd1_rec_node.text = rec_icd1
                    icd2_rec_node = etree.SubElement(child, rec2_icd_tag)
                    icd2_rec_node.text = rec_icd2
                    rec1_keywords_node = etree.SubElement(child, rec1_keywords_tag)
                    rec1_keywords_node.text = fix_keywords(row[rec1_keywords_name])
                    rec2_keywords_node = etree.SubElement(child, rec2_keywords_tag)
                    rec2_keywords_node.text = fix_keywords(row[rec2_keywords_name])
                    if rec_icd1.upper() != rec_icd2.upper():
                        icd = row[adj_icd_name]
                    elif len(rec_icd1) > 0 and len(rec_icd2) > 0 and rec_icd1[0] == rec_icd2[0]:
                        icd = rec_icd1
            else:
                icd = icd1
            if icd_name in row: # Always take the final_icd if it's there
                icd = row[icd_name]
                
            if icd == "":
                print "Error: no ICD code found for " + row[id_name]
                root.remove(child)
                dropped = dropped +1
            #icd = row[adj_icd_name]
            icd_node = etree.SubElement(child, icd_tag)
            icd_node.text = icd

    # Write the xml to file
    adult_file = args.outfile + ".adult"
    child_file = args.outfile + ".child"
    neonate_file = args.outfile + ".neonate"
    etree.ElementTree(adult_root).write(adult_file)
    etree.ElementTree(child_root).write(child_file)
    etree.ElementTree(neonate_root).write(neonate_file)

    # Fix formatting
    subprocess.call(["cleanfile.sh", adult_file, adult_file + ".clean"])
    subprocess.call(["cleanfile.sh", child_file, child_file + ".clean"])
    subprocess.call(["cleanfile.sh", neonate_file, neonate_file + ".clean"])
    data_util.fix_line_breaks(adult_file + ".clean", "adult")
    data_util.fix_line_breaks(child_file + ".clean", "child")
    data_util.fix_line_breaks(neonate_file + ".clean", "neonate")
    
    print "Dropped " + str(dropped) + " records because of missing or mismatched ICD codes"

def fix_keywords(text):
    # Replace | with comma
    return re.sub(r'(?is)\|-', ',', text).decode('utf-8')

def fix_narr(text):
    # Remove backslash
    text = text.replace("\\", "")
    return text.decode('utf-8')

def month_to_num(text):
    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
    if text in abbr_to_num:
        return str(abbr_to_num[text])
    else:
        return "?"

if __name__ == "__main__":main()
