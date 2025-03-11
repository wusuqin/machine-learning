import sys


def initialArray(num_array, i,j):
    sets = set(num_array[i])
    list1=[]
    for k in range(9):
        sets.add(num_array[i][k])
    i_start = int(i/3)*3
    j_start = int(j/3)*3
    for k1 in  range(3):
        for k2 in  range(3):
            sets.add(num_array[i_start+k1][j_start+k2])
    for i in range(9):
        if i+1 not in sets:
            list1.append(i+1)
    return list1
def check(ij, num, dict_num):
    i = int(ij/10)
    j = ij%10
    for key in dict_num.keys():
        i1 = int(key/10)
        j1 = int(key%10)
        if((i==i1 or j==j1 or (int(i/3) == int(i1/3) and int(j/3) == int(j1/3))) and  dict_num.get(key) == num):
            return False
    return True

def calc(maps):
    if(len(maps.keys()) == 1) :
        key = list(maps.keys())[0]
        for i in maps.get(key):
            yield {key:i}
    else:
        maps_new = maps.copy()
        key = list(maps.keys())[0]
        del maps_new[key]
        for dict1 in  calc(maps_new):
            dict2 = dict1.copy()
            for num in  maps.get(key):
                if check(key, num, dict1):
                    dict2.update({key:num})
                    yield dict2

nums = []
for line in sys.stdin:
    a = line.split()
    nums.append([int(i) for i in a])
    if(len(nums) < 9):
        continue
    dict_nums = {}
    for i in range(9):
        row = nums[i]
        for j in range(9):
            if(row[j] == 0):
                list1 = initialArray(nums, i, j)
                dict_nums.update({(i*10+j):list1})
    for dict1 in calc(dict_nums):
        for key in dict1.keys():
            i = int(key/10)
            j = int(key%10)
            nums[i][j] = dict1.get(key)
        break
    for i in range(9):
        print(" ".join([str(s) for s in nums[i]]))
    nums.clear()
    dict_nums.clear()