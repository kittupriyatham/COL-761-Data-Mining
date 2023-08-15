import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from math import log
from functools import partial
import multiprocessing
import sys
import pickle

sorted_pattern_to_index = {}

log2 = lambda x: log(x, 2)
def Unary(x):
    return (x-1)*'0'+'1'

def Binary(x, l = 1):
    s = '{0:0%db}' % l
    return s.format(x)

def Elias_Gamma(x):
    if(x == 0):
        return '0'

    n = 1 + int(log2(x))
    b = x - 2**(int(log2(x)))

    l = int(log2(x))

    return Unary(n) + Binary(b, l)

def process_transaction(transaction_set, sorted_pattern_to_index):
    compressed_set = set()
    for item_st, cod in sorted_pattern_to_index.items():
        if item_st.issubset(transaction_set):
            transaction_set = transaction_set - item_st
            compressed_set.add(sorted_pattern_to_index.get(frozenset(item_st)))
    compressed_set.update(transaction_set)
    return compressed_set


def compress(inputPath,outputPath):
    global sorted_pattern_to_index
    df = pd.read_table(inputPath, sep="\s+",header=None)
    te = TransactionEncoder()
    te_ary = te.fit(df.values.tolist()).transform(df.values.tolist())
    dfr = pd.DataFrame(te_ary, columns=te.columns_)
    gd=fpgrowth(dfr, min_support=0.6,use_colnames=True)
    gd = gd[gd['itemsets'].str.len() > 2]
    gd["frequency"]=gd["support"]*(gd['itemsets'].str.len())
    pattern_to_index = {frozenset(itemset): hex(int(Elias_Gamma(index),2)) for index, itemset in enumerate(gd['itemsets'])}
    list_of_sets = [set(row) for _, row in df.iterrows()]
    sorted_pattern_to_index = dict(sorted(pattern_to_index.items(), key=lambda item: len(item[0]),reverse=True))
    compressed_lst=[]
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    partial_process = partial(process_transaction, sorted_pattern_to_index=sorted_pattern_to_index)

    # Create a multiprocessing pool
    pool = multiprocessing.Pool(num_cores)

    # Parallelize the processing of transaction sets
    compressed_lst = pool.map(partial_process, list_of_sets)
    pool.close()
    pool.join()
    cmp_df=pd.DataFrame(compressed_lst)
    cmp_df.to_csv(outputPath, sep='\t', index=False,header=False)
    with open('sorted_pattern_to_index.pkl', 'wb') as file:
        pickle.dump(sorted_pattern_to_index, file)

def decompress(inputPath,outputPath):
    with open('sorted_pattern_to_index.pkl', 'rb') as file:
        sorted_pattern_to_index = pickle.load(file)
    cf = pd.read_table(inputPath, sep="\s+",header=None)
    print(sorted_pattern_to_index)
    decomp_lst=[]
    rev_indx={str(value): key for key, value in sorted_pattern_to_index.items()}
    list_of_compressed_sets = [set(row) for _, row in cf.iterrows()]
    for comp_set in list_of_compressed_sets:
        decomp_set=set()
        for comp_item in comp_set:
            if comp_item =='nan':
               continue
            elif str(comp_item).startswith("0x"):
               decomp_set.update(rev_indx.get(str(comp_item), []))
            else:
               decomp_set.add(comp_item)
        decomp_lst.append(decomp_set)
    pd.DataFrame(decomp_lst).to_csv(outputPath, sep='\t', index=False,header=False)
           

def compress_main(input_path, output_path):
    global sorted_pattern_to_index
    sorted_pattern_to_index=compress(input_path, output_path)

def decompress_main(input_path, output_path):
    decompress(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py [C/D] input_path output_path")
    elif sys.argv[1] == 'C':
        compress_main(sys.argv[2], sys.argv[3])
        print('Compression complete')
    elif sys.argv[1] == 'D':
        decompress_main(sys.argv[2], sys.argv[3])
        print('Decompression complete')
    elif sys.argv[1] == 'E':
        print('Exiting the program')
        sys.exit(0)
