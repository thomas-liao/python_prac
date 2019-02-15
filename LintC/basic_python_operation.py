# ## documents common operations in python3
#
#
# # 1.
# ord_ = ord('a')
# print(ord_) # 97
# # ------------------------------------------------------------------------------------
#
# # 2  char.isxxx
# print('a'.isdigit()) # false
#
# print('a'.islower()) # need to be all lower
#
# print('aUUU'.isupper()) # need to be all Upper
#
# print('123123asdfxcv'.isalnum()) # is alpha numerical, need to be all alphanumerical
#
# print('     '.isspace())
# print('  '.isspace())
#
# print('abcABC'.isalpha()) # true
# # ------------------------------------------------------------------------------------
#
# # 3 lower/upper conversion
#
# test_str = 'abcd][' # only convert the character while ignoring others
#
# print(test_str.upper()) # ABCD][
#
# print(test_str.lower())
#
# # ------------------------------------------------------------------------------------
#
# # 4 range
#
# for i in range(2, 0, -1): # [start, end), step_size = -1
#     print(i) # print 2, 1
#


#
# ## 5. filer, map and lambda
#
# old_list = [1,2,3,4,5,6,7,8]
# # we wish to get even numbers from old_list
#
# new_list = list(filter(lambda x: x%2 ==0, old_list))
# print(new_list)
#
# # get square
# new_list2 = list(map(lambda x: x * x, old_list))
# print(new_list2)
#
#
# ## 6, sort(key = lambdaxxx)
#
# list = [(1,2,3), (1,2,4), (1,2,5), (1,3,4), (1,3,5)]
# # comparison rule: for (a, b,c) compare a, if equal then compare b, if equal then c
#
# list.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
#
# print(list)


# ## 7, heap
# import heapq
#
# # heapify
# list = [3,2,6,3,2,1,6,4,3,9]
#
# t = heapq.heapify(list)
# print(list)
# # print(t) # None.. so don't do assignment for heapify
#
#
# # heap push
# heapq.heappush(list, 5)
#
# # heatppop
# print(heapq.heappop(list))
#
# # heappushpop
# # this one is quick.. for example, if push element is smaller than original list[0],
# # just return it without having to mess around the list
# print(list) # [2, 2, 3, 3, 5, 6, 6, 4, 3, 9]
#
# heapq.heappushpop(list, -1)
# print(list) # [2, 2, 3, 3, 5, 6, 6, 4, 3, 9] nothing changed
#
# heapq.heappushpop(list, 100)
# print(list) # [2, 3, 3, 3, 5, 6, 6, 4, 100, 9]
#
# # nlargest, nsmallest
# ret = heapq.nlargest(3, list)
# print(ret)
#
# ret2 = heapq.nsmallest(3, list)
# print(ret2)
#
# # merge
# list2 = [5,7,7,8,2,100,333,666]
#
# temp = heapq.merge(list, list2)
# print(temp) # <generator object merge at 0x11834b728>
#
# e = [i for i in temp]
# print(e) # this works
#
# # print(list(temp)) # somehow this doesn't work.. weird
#
# print(list) # [2, 3, 3, 3, 5, 6, 6, 4, 100, 9]
# heapq.heappushpop(list, 0) # [2, 3, 3, 3, 5, 6, 6, 4, 100, 9], so 0 is not added into list
#
#
# heapq.heapreplace(list, 0)
# print(list) # [0, 3, 3, 3, 5, 6, 6, 4, 100, 9],   so 0 is added into list
#





## random
import random
print(random.choice(range(0, 5)))
