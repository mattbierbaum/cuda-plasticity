import bisect

class SortedList(list):
    def __init__(self, *args, **kwargs):
        ret = list.__init__(self, *args, **kwargs)
        self.sort()
        return ret 
 
    def append(self, element):
        bisect.insort(self, element)        

    def find(self, element):
        index = bisect.bisect_left(self, element)
        if index < len(self) and self[index] == element:
            return index
        else:
            return None

    def find_pos_right(self, element):
        index = bisect.bisect_right(self, element)
        return index

    def max(self):
        return self[-1]

    def min(self):
        return self[0]

    def __setitem__(self, index, value):
        self.pop(index)
        self.append(value)

    def remove(self, element):
        ind = self.find(element)
        if ind is not None:
            self.pop(ind)
        else:
            raise ValueError

    def __contains__(self, element):
        index = self.find(element)
        if index is not None:
            return True
        else:
            return False

class LargeSortedList:
    def __init__(self, sorted_list):
        import numpy
        whole = SortedList(sorted_list)
        skip = int(numpy.sqrt(len(whole)))
        self.keys = whole[::skip]
        self.arrs = []
        for i in range(len(self.keys)):
            self.arrs.append(SortedList(whole[i*skip:min((i+1)*skip,len(whole))]))
        # Only need N-1 keys
        self.keys = SortedList(self.keys[1:])

    def Rearrange(self):
        """
        Rearrnge the bins. You need to execute this once in a while for the
        bins to be balanced.
        """
        import numpy
        whole = []
        for arr in self.arrs:
            whole += arr
        skip = int(numpy.sqrt(len(whole)))
        self.keys = whole[::skip]
        self.arrs = []
        for i in range(len(self.keys)):
            self.arrs.append(SortedList(whole[i*skip:min((i+1)*skip,len(whole))]))
        # Only need N-1 keys
        self.keys = SortedList(self.keys[1:])

    def append(self, element):
        bin_index = self.keys.find_pos_right(element)
        self.arrs[bin_index].append(element)

    def find(self, element):
        bin_index = self.keys.find_pos_right(element)
        index = self.arrs[bin_index].find(element)
        if index is not None:
            return (bin_index, index)
        else:
            return None 

    def max(self):
        return self.arrs[-1][-1]

    def min(self):
        return self.arrs[0][0]

    def __setitem__(self, index, value):
        self.arrs[index[0]].pop(index[1])
        self.append(value)

    def __getitem__(self, index):
        return self.arrs[index[0]][index[1]]

    def remove(self, element):
        ind = self.find(element)
        if ind is not None:
            self.arrs[ind[0]].pop(ind[1])
        else:
            raise ValueError

    def __contains__(self, element):
        index = self.find(element)
        if index is not None:
            return True
        else:
            return False

    def pop(self, index):
        self.arrs[index[0]].pop(index[1])
        """
        Invoke rearrangement of the bins when one becomes empty.
        This is crucial especially for max/min to work since they
        explicitly look for a specific element which may not exist
        if the bins are empty.
        """
        if len(self.arrs[index[0]]) == 0:
            self.Rearrange()

class SortedIndexPreservingList(list):
    def __init__(self, *args, **kwargs):
        ret = list.__init__(self, *args, **kwargs)
        self.sortedlist = SortedList([(v,i) for i,v in enumerate(self)])
        return ret 
 
    def append(self, element):
        list.append(self, element)
        bisect.insort(self.sortedlist, (element,len(self)))        

    def argmax(self):
        return self.sortedlist[-1][1]

    def argmin(self):
        return self.sortedlist[0][1]

    def max(self):
        return self.sortedlist[-1][0]

    def min(self):
        return self.sortedlist[0][0]

    def __setitem__(self, index, value):
        original_value = self[index]
        sorted_index = self.sortedlist.find((original_value, index))
        self.sortedlist[sorted_index] = (value, index)
        list.__setitem__(self, index, value)

class ValueSortedDict(dict):
    def __init__(self, *args, **kwargs):
        ret = dict.__init__(self, *args, **kwargs)
        self.sortedlist = SortedList([(v,i) for i,v in self.items()])
        return ret 
 
    def argmax(self):
        return self.sortedlist.max()[1]

    def argmin(self):
        return self.sortedlist.min()[1]

    def max(self):
        return self.sortedlist.max()[0]

    def min(self):
        return self.sortedlist.min()[0]

    def __setitem__(self, index, value):
        if index in self:
            sorted_index = self.sortedlist.find((self[index], index))
            self.sortedlist[sorted_index] = (value, index)
        else:
            self.sortedlist.append((value, index))
        dict.__setitem__(self, index, value)

    def pop(self, index):
        sorted_index = self.sortedlist.find((self[index], index))
        self.sortedlist.pop(sorted_index)
        return dict.pop(self, index)
       
class LargeValueSortedDict(ValueSortedDict):
    def __init__(self, *args, **kwargs):
        ret = dict.__init__(self, *args, **kwargs)
        self.sortedlist = LargeSortedList([(v,i) for i,v in self.items()])
        return ret 
 
