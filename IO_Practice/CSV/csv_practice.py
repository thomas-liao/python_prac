# https://www.youtube.com/watch?v=q5uM4VKywbA
# open(context), define reader, (open(context) define writer), writer.writerow(line in reader)


# part 1, standard..
# import csv
#
# with open('name.csv', 'r') as read_file:
#     csv_reader = csv.reader(read_file)
#
#     next(csv_reader)
#
#     with open('new_names.csv', 'w') as write_file:
#         csv_writer = csv.writer(write_file, delimiter='\t')
#
#         for line in csv_reader:
#             csv_writer.writerow(line)





# part2, dictionary reader... author prefer this one


import csv
with open('name.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    with open('new_names_dict.csv', 'w') as new_file:
        # fieldNames = ['first_name', 'last_name', 'email']
        fieldNames = ['first_name', 'last_name']

        csv_writer = csv.DictWriter(new_file, fieldnames=fieldNames, delimiter='\t')

        csv_writer.writeheader() # write fieldname in first row... you usually need to do that

        for line in csv_reader:
            del line['email']
            csv_writer.writerow(line)





