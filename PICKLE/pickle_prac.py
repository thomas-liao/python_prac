import pickle

# a_dict = {'da':111, 2:[2,3,1,4], '23':{1:2, 'd':'sad'}}
#
# file = open('pickle_example.pickle', 'wb')
#
# pickle.dump(a_dict, file)
#
# file.close()


# suppose later you want to play with saved pickle file
#
# with open('pickle_example.pickle', 'rb') as f:
#     a_dict1 = pickle.load(f)
#
# print(a_dict1)
#
import matplotlib.pyplot as plt

keys = ['a', 'b','c','d','e'] # 5 scenes
#
# with open('test_example.pickle', 'wb') as f:
#     pickle.dump({'a':{'TPR_list':[], 'FPR_list': []}}, f)

with open('test_example.pickle', 'rb') as f:
    det = pickle.load(f)


# det['b'] = {'TPR_list:':[], 'FPR_list:':[]}
#
# with open('test_example.pickle', 'wb') as fout:
#     pickle.dump(det, fout)




print(det)