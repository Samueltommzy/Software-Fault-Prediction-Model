# Pseudo code

#single arff file
#1. For each line in the file, if line starts with @attribute
  #readline to string and split string by space
  #Push splittedValue[1] to the column title array

from os import listdir, path
import csv

#retrieve all datasets from the raw_data directory
raw_datasets = [dataset for dataset in listdir('./raw_data') if dataset.endswith('.arff')]
print(raw_datasets)

#loop through the datasets and generate a csv file
# for dataset in raw_datasets:
#   with open(dataset, 'r') as file:
#     file_content = file.readlines()

with open('./raw_data/cm1.arff', 'r') as file:
  file_content = file.readlines()
  file_name, file_extension = path.splitext(file.name)
  csv_data = []
  header = []
  data = False
   
  for content in file_content:
     #append @data before the dataset values
    if not content.startswith('@attribute'):
      content = '@data' + ' ' +  content
    if content.startswith('@attribute'):
      column_name = content.split(' ')[1]
      header.append(column_name)
    elif "@data" in content:
      csv_data.append(content.split(' ')[1])
    else:
      pass
  # print(file_name, file_extension)
 
  final_data = [d.split(',') for d in csv_data]
  # print(header,final_data)
  with open(file_name+'.csv', 'w') as csv_output:
    wr = csv.writer(csv_output)
    wr.writerow(header)
    wr.writerows(final_data)
  csv_output.close()