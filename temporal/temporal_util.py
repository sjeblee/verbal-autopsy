#!/usr/bin/python3
# Temporal utility functions

import dateutil

none_label = "NONE"

def is_date(text):
    try:
        dateutil.parser.parse(text)
        return True
    except ValueError:
        return False

def flip_relations(id1, id2, rel_type):
    if rel_type == 'AFTER':
        return id2, id1, 'BEFORE'
    elif rel_type == 'TERMINATES':
        return id2, id1, 'ENDS-ON'
    elif rel_type == 'INITIATES':
        return id2, id1, 'BEGINS-ON'
    else:
        return id1, id2, rel_type

def map_rel_type(rel_type, relation_set):
    rel_type = rel_type.upper()
    if relation_set == 'exact':
        return rel_type
    if relation_set == 'binary':
        if rel_type == none_label:
            return 0
        else:
            return 1
    elif relation_set == 'simple':
        if rel_type == none_label:
            return rel_type
        elif rel_type in ['BEFORE', 'IBEFORE', 'BEFORE/OVERLAP', 'ENDS-ON']:
            return 'BEFORE'
        elif rel_type in ['AFTER', 'IAFTER', 'BEGINS-ON']:
            return 'AFTER'
        else:
            return 'OVERLAP'
    elif relation_set == 'contains':
        if rel_type == 'CONTAINS':
            return rel_type
        else:
            return none_label

def print_time(t):
    unit = "s"
    if t>60:
        t = t/60
        unit = "mins"
    if t>60:
        t = t/60
        unit = "hours"
    print("time:", str(t), unit)
    return str(t) + " " + unit

'''
Convert a text representing a date to a date object
text: the string to convert
'''
def to_date(text):
    try:
        date = dateutil.parser.parse(text)
        return date
    except ValueError:
        return None
