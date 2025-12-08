import json, csv, sys


def import_json_file(filename) -> list[list[int|float|bool]]:
    f = open(filename, 'r')
    file = json.load(f)
    f.close()
    return file

def import_csv_file(filename) -> list[int|str]:
    f = open(filename, 'r')
    file = csv.reader(f, delimiter=',')
    data = []
    for row in file:
        data.append(row)
    return data

def process_campaign_reqs(campaign_reqs_file):
    rows = len(campaign_reqs_file)
    cols = len(campaign_reqs_file[0])-1
    campaign_reqs = [[0 for j in range(cols-2)] for i in range(rows-1)]
    i = 0
    for row in campaign_reqs_file:
        if i > 0:  # Row 0 is headers
            for j in range(cols):
                if j > 1 and j < cols-3 and row[j] != '':  # Column 0 and 1 are # and name
                    campaign_reqs[i-1][j-2] = (float(row[j]))

                if j == cols-1 or j == cols-2 or j == cols-3:  # Last columns are sequence constraints
                    # print(i-1,j-2,cols, row[j])
                    campaign_reqs[i - 1][j - 2] = row[j]
        i += 1

    return campaign_reqs
