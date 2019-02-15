import csv

with open('def54_training_log.log', 'r') as log_file:
    temp = []
    for line in log_file:
        if line[:8] == "Training":
            idx = line.rfind(': ')
            # print(float(line[idx + 1:]))
            temp.append(float(line[idx + 1:]))

    with open('Training_loss.csv', 'w') as loss_file:
        fieldNames = ['iter_count', 'loss']
        csv_writer = csv.DictWriter(loss_file, fieldnames=fieldNames, delimiter=',')
        csv_writer.writeheader()
        for i in range(len(temp)):
            idx = 100 * (i+1)
            loss = temp[i]
            temp_dict = {}
            temp_dict['loss'] = loss
            temp_dict['iter_count'] = idx
            csv_writer.writerow(temp_dict)


# import csv
#
# with open('name.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#
#     with open('new_names_dict.csv', 'w') as new_file:
#         # fieldNames = ['first_name', 'last_name', 'email']
#         fieldNames = ['first_name', 'last_name']
#
#         csv_writer = csv.DictWriter(new_file, fieldnames=fieldNames, delimiter='\t')
#
#         csv_writer.writeheader()  # write fieldname in first row... you usually need to do that
#
#         for line in csv_reader:
#             del line['email']
#             csv_writer.writerow(line)