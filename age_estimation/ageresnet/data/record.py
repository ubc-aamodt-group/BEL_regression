from pathlib import Path
import csv

def appendToCsv(file, labels, data):
    try:
        with open(file, 'a', newline='') as fd:
            rd = csv.reader(fd, delimiter=',',
                            quotechar='|')
            wr = csv.writer(fd, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(data)
    except:
        with open(file, 'a', newline='') as fd:
            wr = csv.writer(fd, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            wr.writerow(labels)
            wr.writerow(data)


