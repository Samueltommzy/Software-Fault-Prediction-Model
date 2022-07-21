from os import listdir, path
import csv

def arff_to_csv(file):
  file_name = file.split('.')[0]
  with open('./raw_data/'+file, 'r') as arff_file:
    file_content = arff_file.readlines()
    rows = []
    columns = []


    for content in file_content:
      #append @data before the dataset values which will serve as rows in the csv file
      if not content.startswith('@attribute'):
        content = '@data' + ' ' +  content
      #attributes serve as column in the csv file
      if content.startswith('@attribute'):
        column_name = content.split(' ')[1]
        columns.append(column_name)
      elif content.startswith('@data'):
        rows.append(content.split(' ')[1])
      else:
        pass
    row_data = [row.split(',') for row in rows]
    #write to a csv file in a new csv_data folder
    with open('./csv_data/'+file_name+'.csv', 'w') as csv_output:
      wr = csv.writer(csv_output)
      wr.writerow(columns)
      wr.writerows(row_data)
    csv_output.close()

if __name__ == "__main__":
  #retrieve all datasets from the raw_data directory
  raw_datasets = [dataset for dataset in listdir('./raw_data') if dataset.endswith('.arff')]
  for file in raw_datasets:
    arff_to_csv(file)
