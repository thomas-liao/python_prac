import csv

with open('/Users/admin/Desktop/test_log/run_log', 'r') as log_file:
    avg = []
    all_ = []
    line = log_file.readline()
    while line:
        if line[:13] == 'Avg val accur':

            avg_accur = float( ''.join(v for v in line if v.isdigit() or v =='.'))
            avg.append(avg_accur)
            next_line = log_file.readline()
            all_accur = eval(next_line)
            all_.append(all_accur)
        line = log_file.readline()
    log_file.close()

    # error proof
    assert len(avg) == len(all_)
    for i in range(len(avg)):
        all_[i]['avg_accur'] = avg[i]
        # print(all_[i])

    with open('accur_log.csv', 'w') as accur_file:
        fieldnames = []
        for k in all_[0].keys():
            fieldnames.append(k)
        fieldnames.remove('avg_accur')
        fieldnames[0:0] = ['avg_accur'] # must have [] otherwise malfunctioning... not as expected
        # print(fieldnames)
        csv_writer = csv.DictWriter(accur_file, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()
        for i in range(len(all_)):
            # print('Sanity check')
            # print(all_[i])
            csv_writer.writerow(all_[i])
    # for line in log_file:



    # with open('loss_log', 'w') as loss_file:
    #     fieldNames = ['iter_count', 'loss']
    #     csv_writer = csv.DictWriter(loss_file, fieldnames=fieldNames, delimiter=',')
    #     csv_writer.writeheader()
    #     for i in range(len(temp)):
    #         idx = 100 * (i+1)
    #         loss = temp[i]
    #         temp_dict = {}
    #         temp_dict['loss'] = loss
    #         temp_dict['iter_count'] = idx
    #         csv_writer.writerow(temp_dict)







#
#
#
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