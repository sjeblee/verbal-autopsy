#!/usr/bin/python3
# Temporal utility functions

def map_rel_type(rel_type, relation_set):
    if relation_set == 'exact':
        return rel_type
    if relation_set == 'binary':
        if rel_type == none_label:
            return 0
        else:
            return 1
    elif relation_set == 'simple':
        if rel_type == 'NONE':
            return rel_type
        elif rel_type in ['BEFORE', 'IBEFORE']:
            return 'BEFORE'
        elif rel_type in ['AFTER', 'IAFTER']:
            return 'AFTER'
        else:
            return 'OVERLAP'

def print_time(t):
    unit = "s"
    if t>60:
        t = t/60
        unit = "mins"
    if t>60:
        t = t/60
        unit = "hours"
    return str(t) + " " + unit
