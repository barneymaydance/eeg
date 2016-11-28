import csv
import numpy
import marshal as m
from datetime import datetime

filename = "indra_mids_5_15/subject-metadata.csv"
male=[]
female=[]
with open(filename, "rU") as csvfile:
    reader = csv.reader(csvfile,dialect=csv.excel_tab)
    for index, col in enumerate(reader):
        if index == 0:
            continue
        gender=col[0].split(",")[5]
        id=col[0].split(",")[0]
        if gender=="f":
            female.append(int(id))
        else:
            male.append(int(id))

print male
print female