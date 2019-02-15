import h5py

# essence of 3 basic type of items in h5py:
# file, group, dataset. their names are used as access key

# h5py structure is similar to the file system directory tree
# top level of HDF5 tree: a file
## file may contain groups and datasets
## each group may contain other groups and datasets
## each dataset contains the data objects, which in most cases can be associated with NumPy types

# advantage: flexible, efficient storage and I/O



import numpy as np
import h5py

### create file
# matrix1 = np.random.random(size=(1000, 1000))
# matrix2 = np.random.random(size=(10000, 100))
#
# with h5py.File('temp_h5py.h5', 'w') as hdf:
#     hdf.create_dataset('dataset1', data=matrix1)
#     hdf.create_dataset('dataset2', data=matrix2)

### load file

#with h5py.File('temp_h5py.h5', 'r') as hdf:
#    ls = list(hdf.keys())
#    print("List of datasets in the file:\n", ls)
#    data = hdf.get('dataset1')
#    dataset1 = np.array(data)
#    print("shape of dataset1: \n", dataset1.shape)



## create groups
#
# matrix1 = np.random.random(size=(1000, 1000))
# matrix2 = np.random.random(size=(10000, 10))
# matrix3 = np.random.random(size=(1000, 1000))
# matrix4 = np.random.random(size=(1000, 100))
#
# with h5py.File('hdf5_group.h5', 'w') as hdf:
#     G1 = hdf.create_group("Group1")
#     G1.create_dataset('dataset1', data=matrix1) # instead of hdf.create_dataset
#     G1.create_dataset('dataset4', data=matrix4)
#
#     G21 = hdf.create_group('Group2/SubGroup1')
#     G21.create_dataset('dataset3', data=matrix3)
#
#     G22 = hdf.create_group('Group2/SubGroup2')
#     G22.create_dataset('dataset2', data = matrix2)
#
#
# with h5py.File('hdf5_group.h5') as hdf:
#     base_items = list(hdf.items())
#     print("Items in the base directory: ", base_items)
#     ## res
#     ## Items in the base directory:  [('Group1', <HDF5 group "/Group1" (2 members)>),
#     ## ('Group2', <HDF5 group "/Group2" (2 members)>)]
#
#
#     # take a look into G1
#     G1 = hdf.get('Group1')
#     G1_items = list(G1.items())
#     print('Items in G1:', G1_items)
#     ## res
#     ## Items in G1: [('dataset1', <HDF5 dataset "dataset1": shape (1000, 1000), type "<f8">),
#     ##  ('dataset4', <HDF5 dataset "dataset4": shape (1000, 100), type "<f8">)]
#     dataset4 = np.array(G1.get('dataset4'))
#     print("dataset4 shape: ", dataset4.shape)
#
#
#     # take a look into G2
#     G2 = hdf.get('Group2')
#     G2_items = list(G2.items())
#     print("Items in G2: ", G2_items)
#
#     # dip into G2
#     G21 = G2.get('SubGroup1')
#     G21_items = list(G21.items())
#     print('G21_itemsL',  G21_items)
#
#     dataset3 = np.array(G21.get('dataset3'))
#     print("dataset3 shape: ", dataset3.shape)


#

### nice little utility function to traverse keys of datasets
# def extract(name, node):
#     if isinstance(node, h5py.Dataset):
#         dd[name] = node
#     return None
#
#
# dd = {}
# with h5py.File('hdf5_group.h5', 'r') as f:
#     f.visititems(extract)
#     print(dd)
#     temp = list(dd.keys())
#     print(temp)
#


### more simple and naive: verbosely print out (name, node) pair
# def temp(a, b):
#     print("--> name: ", a)
#     print("    node: ", b)
#
# dd = {}
#
# with h5py.File('hdf5_group.h5', 'r') as f:
#     f.visititems(temp)






### finally compression
# with h5py.File('hdf5_group_compressed.h5', 'w') as hdf:
#     G1 = hdf.create_group('Group1')
#     G1.create_dataset('dataset1', data=np.arange(10), compression='gzip', compression_opts=9)




### also you may add attributes: like, version = 1.1, class = data matrix etc

# with h5py.File('hdf5_with_attributes.h5', 'w') as hdf:
#     dataset1 = hdf.create_dataset('dataset1', data=np.arange(10000))
#
#     # set attributes
#     dataset1.attrs['CLASS'] = 'DATA MATRIX'
#     dataset1.attrs['VERSION'] = '1.1'


# read and see how it works
#
# with h5py.File('hdf5_with_attributes.h5', 'r') as hdf:
#     ls = list(hdf.keys())
#     print('List of datasets in the file:\n', ls)
#
#     data = hdf.get('dataset1')
#     data_np = np.array(data)
#
#     k = list(data.attrs.keys())
#     v = list(data.attrs.values())
#     print(k) # ['CLASS', 'VERSION']
#     print(v)
#
#



#### interaction with pandas - 9, 10 / 10: incomplete
##  https://www.youtube.com/watch?v=EsYBriqMv0U&index=9&list=PLea0WJq13cnB_ORdGzEkPlZEN20TSt6Lx




####################################################################################################
## now try to create a dataset as you need

### create file
# matrix1 = np.random.random(size=(1000, 1000))
# matrix2 = np.random.random(size=(10000, 100))
#

# with h5py.File('emb.h5', 'w') as hdf:
#
#     # hdf.create_dataset('seq1', data={1:[1,2,3,4,5,128]}) # doesn't work
#     # hdf.create_dataset('seq1', data={1:[1,2,3,4,5]}) #
#     # Todo: all embeddings in seq referenced by dictionary key of frame_id)
#
#     grp = hdf.create_group('tf_triplet_embedding')
#
#
#     data_to_save = {1:[1,2,3,4,5,6], 2:[3,4,5,6,5,4,2], 3:[4,4,4,4,3,3,2,1]}
#
#     for k, v in data_to_save.items():
#         grp.create_dataset(str(k), data=np.array(v))
#
#
# def extract(a, b):
#     print("---> name: ", a, "Group" if isinstance(b, h5py.Group) else "Dataset")
#     print("     node: ", np.array(b))
#
#
# with h5py.File('emb.h5', 'r') as f:
#     f.visititems(extract)
#
#
# # access value seq_id/frame_id
# with h5py.File('emb.h5', 'r') as hdf:
#     res = hdf.get('tf_triplet_embedding/1')
#     print(np.array(res))
####################################################################################################


# although above code could work... it is just kind of ugly, let's try to use pandas to refine it
#
# import pandas as pd
#
# # hdf = pd.HDFStore('emb.h5')
# hdf = pd.HDFStore('h5_pd_prac.h5')
# df1 = pd.read_csv('/Users/admin/Desktop/JHU_18_Spring/Data Mining/Final_Project/treatmentGroup.csv')
#
# hdf.put("DF1", df1, format='table', data_columns=True)
#


# another way is just to use json
# import json
#
# temp = json.dumps({1:[1,2,3,4,5,6], 2:[3,4,5,6,5,4,2], 3:[4,4,4,4,3,3,2,1]})
#
#
# print(json.loads(temp))
# print(type(json.loads(temp)))



###### final techincal details: let's do this

# dummy datum

import json
# seq1: fm1, fm2
seq1_fm1 = np.random.randn(128).tolist()
seq1_fm2 = np.random.randn(128).tolist()

# seq2: fm1, fm2
seq2_fm1 = np.random.randn(128).tolist()
seq2_fm2 = np.random.randn(128).tolist()

with h5py.File('final_version.h5', 'w') as hdf:
    grp = hdf.create_group('triplet_emb_tf')
    temp_dict1 = {1: seq1_fm1, 2:seq2_fm2}
    temp_json1 = json.dumps(temp_dict1)
    temp_dict2 = {1:seq2_fm1, 2:seq2_fm2}
    temp_json2 = json.dumps(temp_dict2)

    print(type(temp_json2)) # str
    check = np.array(temp_json2) # np array... magic
    print(type(check)) #  np array... magic

    grp.create_dataset('seq1', data=temp_json1)
    grp.create_dataset('seq2', data=temp_json2)


#
# def extract(a, b):
#     print("---> name: ", a, "Group" if isinstance(b, h5py.Group) else "Dataset")
#     print("     node: ", np.array(b))
#
#
# with h5py.File('final_version.h5', 'r') as f:
#     f.visititems(extract)


