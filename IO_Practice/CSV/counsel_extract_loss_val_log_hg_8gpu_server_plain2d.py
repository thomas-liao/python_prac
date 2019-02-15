import csv

with open('/Users/admin/PycharmProjects/python_prac/IO_Practice/CSV/train_log_hg_8gpu_server.log', 'r') as log_file:
    class0_loss = []
    class1_loss = []
    class2_loss = []
    class0_all_accur = []
    class1_all_accur = []
    class2_all_accur = []
    class0_avg_accur = []
    class1_avg_accur = []
    class2_avg_accur = []
    iters = []
    line = log_file.readline()
    while line:
        if line[:10] == 'Validation':
            iters.append(''.join(v for v in line if v.isdigit()))
            line = log_file.readline()
            line = log_file.readline()
            # class 0 ce loss
            idx = line.rfind(': ')
            class0_loss.append(float(line[idx+1:]))
            line = log_file.readline()
            line = log_file.readline() # class1 ce loss
            idx = line.rfind(': ')
            class1_loss.append(float(line[idx+1:]))
            line = log_file.readline()
            line = log_file.readline() # class2 ce loss
            idx = line.rfind(': ')
            class2_loss.append(float(line[idx + 1:]))
            line = log_file.readline()
            line = log_file.readline() # class0 avg accur
            idx = line.rfind(': ')
            # print((line[idx + 1:]))
            class0_avg_accur.append(float(line[idx + 1:]))
    #         #
            line = log_file.readline()
            line = log_file.readline()
            line = log_file.readline() # class0 all accuracy
            line = line.replace('\n', '')
            line_str = line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','

            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','

            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','

            line_str = line_str.replace('  ',',')
            line_str = line_str.replace(' ', ',')
            line_str = line_str.replace(',,', ',')
            line_str = line_str.replace(',,', ',')
            class0_all_accur.append(eval(line_str[1:-1])) # append all accuracy

            line = log_file.readline()
            line = log_file.readline() # class1 avg accuracy
            idx = line.rfind(': ')
            class1_avg_accur.append(float(line[idx + 1:]))

            line = log_file.readline()
            line = log_file.readline()
            line = log_file.readline()  # class1 all accuracy
            line = line.replace('\n', '')
            line_str = line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','

            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','

            line_str = line_str.replace('  ', ',')
            line_str = line_str.replace(' ', ',')
            line_str = line_str.replace(',,', ',')
            line_str = line_str.replace(',,', ',')
            class1_all_accur.append(eval(line_str[1:-1]))

            # class 2
            line = log_file.readline()
            line = log_file.readline()  # class2 avg accuracy
            idx = line.rfind(': ')
            class2_avg_accur.append(float(line[idx + 1:]))

            line = log_file.readline()
            line = log_file.readline()
            line = log_file.readline()  # class1 all accuracy
            line = line.replace('\n', '')
            line_str = line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','
            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','

            line = log_file.readline()
            line = line.replace('\n', '')
            line_str = line_str + line + ','

            line_str = line_str.replace('  ', ',')
            line_str = line_str.replace(' ', ',')
            line_str = line_str.replace(',,', ',')
            line_str = line_str.replace(',,', ',')
            class2_all_accur.append(eval(line_str[1:-1]))
        line = log_file.readline()
    log_file.close()
    # build up a giant dictionary
    temp = [] # write loss and avg accuracy
    for i in range(len(class0_loss)):
        all_ = {}
        all_['steps'] = iters[i]
        all_['c0_loss'] = class0_loss[i]
        all_['c1_loss'] = class1_loss[i]
        all_['c2_loss'] = class2_loss[i]
        all_['c0_avg_accur'] = class0_avg_accur[i]
        all_['c1_avg_accur'] = class1_avg_accur[i]
        all_['c2_avg_accur'] = class2_avg_accur[i]
        temp.append(all_.copy())

    with open('hg_car_plain2d_200k_val_log.csv', 'w') as accur_file:
        fieldnames = ['steps', 'c0_loss', 'c1_loss', 'c2_loss', 'c0_avg_accur','c1_avg_accur', 'c2_avg_accur']
        csv_writer = csv.DictWriter(accur_file, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()
        for i in range(len(temp)):
            csv_writer.writerow(temp[i])

    with open('hg_car_plain2d_200k_val_log_all_0.csv', 'w') as accur_file2:
        name_list_0 = []
        for i in range(36):
            name_list_0.append('{}_c0'.format(i))
        temp = []
        for i in range(len(class0_loss)):
            all_ = {}
            all_['steps'] = iters[i]
            for j in range(36):
                all_[name_list_0[j]] = class0_all_accur[i][j]
            temp.append(all_)
        fieldnames = []
        fieldnames.append('steps')
        fieldnames.extend(name_list_0)
        csv_writer = csv.DictWriter(accur_file2, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()
        for i in range(len(class0_loss)):

            csv_writer.writerow(temp[i])


    with open('hg_car_plain2d_200k_val_log_all_1.csv', 'w') as accur_file2:
        name_list_1 = []
        for i in range(36):
            name_list_1.append('{}_c1'.format(i))
        temp = []
        for i in range(len(class0_loss)):
            all_ = {}
            all_['steps'] = iters[i]
            for j in range(36):
                all_[name_list_1[j]] = class1_all_accur[i][j]
            temp.append(all_)
        fieldnames = []
        fieldnames.append('steps')
        fieldnames.extend(name_list_1)
        csv_writer = csv.DictWriter(accur_file2, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()
        for i in range(len(class1_loss)):

            csv_writer.writerow(temp[i])


    with open('hg_car_plain2d_200k_val_log_all_2.csv', 'w') as accur_file2:
        name_list_2 = []
        for i in range(36):
            name_list_2.append('{}_c2'.format(i))
        temp = []
        for i in range(len(class2_loss)):
            all_ = {}
            all_['steps'] = iters[i]
            for j in range(36):
                all_[name_list_2[j]] = class2_all_accur[i][j]
            temp.append(all_)
        fieldnames = []
        fieldnames.append('steps')
        fieldnames.extend(name_list_2)
        csv_writer = csv.DictWriter(accur_file2, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()
        for i in range(len(class2_loss)):

            csv_writer.writerow(temp[i])


####


    #
    # with open('accur_log.csv', 'w') as accur_file:
    #     fieldnames = []
    #     for k in all_[0].keys():
    #         fieldnames.append(k)
    #     fieldnames.remove('avg_accur')
    #     fieldnames[0:0] = ['avg_accur'] # must have [] otherwise malfunctioning... not as expected
    #     # print(fieldnames)
    #     csv_writer = csv.DictWriter(accur_file, fieldnames=fieldnames, delimiter=',')
    #     csv_writer.writeheader()
    #     for i in range(len(all_)):
    #         csv_writer.writerow(all_[i])
    #








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