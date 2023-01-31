import numpy as np
import json
import math
import matplotlib.pyplot as plt
from itertools import product
from tabulate import tabulate
from scipy import stats as st
import sys


file_name = sys.argv[1]

with open(file_name, "r") as file:
    ex = json.load(file)
    print(f'SHOWING RESULTS FOR DATA SET: {file.name} \n')

# Convert the data to an array to create the trace
iterable = (p for p in ex.values())
payloads = np.sort(np.fromiter(iterable, int))
print('-------- GENERAL INFORMATION -------- \n')
print(f'Number of received messages: {payloads.size}')

# To create the trace, we compare the messages that should have arrived with those that actually arrived
expected = np.arange(start=payloads[0],stop=payloads[payloads.size-1]+1,dtype=int)
print(f'Number of expected messages: {expected.size}')
# The trace must be the same size as the total number of sent messages
trace = np.zeros(expected.size, dtype=int)

for i in range(0,expected.size):
    if expected[i] not in payloads:
        trace[i] = 1

"""
We create an array of counters for each time we find a loss in the trace
"""

loss_lengths = []

"""
We go through the trace and each time we find a loss, 
a counter is initialized and grows until a successfully received message is found.
Then, we append the counter to loss_lengths
"""
def count_loss(trc,index,lengths):
    i = index
    count = 0
    while trc[i] == 1 and i < trc.size-1:
        count += 1
        i += 1
    #print(f'Loss count: {count}')
    lengths.append(count)
    return i, lengths

def lengths_for_losses(trc,lossy_lengths):
    ind = 0
    while ind < trc.size:
        if trc[ind] == 1:
            ind, lossy_lenghts = count_loss(trace,ind,lossy_lengths)
        else: ind +=1
    return lossy_lenghts

loss_lengths = lengths_for_losses(trace,loss_lengths)
"""
Calculate the mean and standard deviation of the losses lengths in the trace
"""
mean = np.mean(loss_lengths)
std = np.std(loss_lengths)
loss_rate = np.mean(trace)

print(f'Loss lengths mean: {mean}')
print(f'Standard deviation: {std}')
print(f'Loss rate: {loss_rate}')

"""
Calculate the change-of-state constant C as the mean + standard deviation of the losses lengths
"""
c = mean + std

print(f'Change-of-state constant C: {c} \n')


samples = [i for i in range(500,trace.size,math.floor((trace.size)/2500))]
std_deviations = []

for sample_len in samples:
    losses_lens = []
    sample = np.array(trace[0:sample_len])
    std = np.std(sample)
    std_deviations.append(std)
#print(f'Standard deviations: {std_deviations}')

plt.plot(samples,std_deviations)
plt.ylim(-0.05,0.55)
plt.title('Standard deviations per sample length')
plt.xlabel('Sample length')
plt.ylabel('\u03C3')
plt.show()


lossy_trace = np.array([])
error_free_lengths = []
lossy_state_lengths = []

"""
Now it is time to create the lossy trace

For this, we will create a subroutine that extracts the loss states
and concatenates them in lossy_trace
"""

# This function counts the number of consecutive zeros in a given trace
def zero_counter(trc,index):
    count = 0
    while trc[index] == 0 and index < trc.size-1:
        count += 1
        index += 1
    return count


"""
The following code creates the lossy trace as follows:

1.- Initialized two indices to delimit the start and end of a segment in the trace
2.- Both indices are increased until a loss is found
3.- Then, only the "end" index is increased until a zero is found and we proceed to count 
the number of following zeros
4.- If the number of zeros is less than C, the end-of-segment index is updated and we continue iterating
5.- If the number of zeros is greater than C, the segment trace[start:end] is added to lossy_trace

We will also create the error-free trace to compare the lengths of the lossy and error-free states
"""

start = 0
end = 0
while start < trace.size-1 and end < trace.size-1:
    if trace[start] == 0 and start == end:
        count = zero_counter(trace,start)
        error_free_lengths.append(count)
        start += count
        end += count
        continue
    elif trace[start] == 1:
        # Updates end
        end += 1
        # If we find a zero, start to count
        if trace[end] == 0:
            count = zero_counter(trace,end)
            # If the number if consecutive zeros is greater than C, start index is updated
            if count < c:
                end += count
                continue
            else: 
                error_free_lengths.append(count)
                lossy_state = trace[start:end]
                lossy_state_lengths.append(lossy_state.size)
                lossy_trace = np.concatenate((lossy_trace, np.array(lossy_state)))
                end += count
                start = end

# Prints information and graphs about the error-free and lossy traces
print('-------- ERROR-FREE TRACE INFORMATION -------- \n')
print(f'Minimum error-free length: {np.min(error_free_lengths)}')
print(f'Maximum error-free length: {np.max(error_free_lengths)}')
print(f'Mean: {np.mean(error_free_lengths)}')
print(f'Mode: {st.mode(error_free_lengths,axis=0,keepdims=False)}')
print(f'Median: {np.median(error_free_lengths)}\n')
plt.hist(error_free_lengths,bins=25)
plt.title('Lengths of error-free segments')
plt.ylabel('Frequency')
plt.xlabel('Error-free lengths')
plt.grid(True)
plt.show()

print('-------- LOSSY TRACE INFORMATION -------- \n')
print(f'Lossy trace length: {lossy_trace.size}')
print(f'Minimum loss state length: {np.min(lossy_state_lengths)}')
print(f'Maximum loss state length: {np.max(lossy_state_lengths)}')
print(f'Mean: {np.mean(lossy_state_lengths)}')
print(f'Mode: {st.mode(lossy_state_lengths,axis=0,keepdims=False)}')
print(f'Median: {np.median(lossy_state_lengths)} \n')
plt.hist(lossy_state_lengths,bins=25)
plt.title('Lengths of lossy segments')
plt.ylabel('Frequency')
plt.xlabel('Loss lengths')
plt.grid(True)
plt.show()

"""
The runs test is to test the stationarity of a given trace.
In this case, we will prove that the lossy trace is stationary (and therefore it could be
modeled with a Discrete-Time Markov Chain or DTMC)

The test is summarized as follows:

1.- Define a run as a number of consecutive ones (error burst)
2.- Divide the trace into segments of equal lengths
3.- Compute the lengths of runs in each segment
4.- Count the number of runs of length above and below the median value for run lengths in the trace
5.- Plot a histogram for the number of runs

For a stationary trace, the number of runs distribution between 0.05 and 0.95 cut-offs
will be close to 90 percents.
"""
def runs_test(trc):
    print('-------- RUNS TEST INFORMATION -------- \n')
    # We choose the size in which the trace is going to be partitioned
    window_size = 50
    partitions = math.floor((trc.size)/window_size)
    print(f'Number of partitions: {partitions}')
    # The trace is divided into equal parts
    trc_partitioned = np.array_split(trc,partitions)

    runs = np.array([])
    for prtn in trc_partitioned:
        i = 0
        sub_runs = []
        while i < prtn.size-1:
            if prtn[i] == 1:
                i, sub_runs = count_loss(prtn,i,sub_runs)
            else: i += 1
        runs = np.concatenate((runs,np.array(sub_runs)))
    
    median = np.median(runs)
    runs_above = np.where(runs > median)
    print(f'Number of runs above median: {runs_above[0].size}')
    runs_below = np.where(runs < median)
    print(f'Number of runs below median: {runs_below[0].size} \n')

    print(f'Number of runs: {runs.size}')
    cut_off_5 = np.max(runs) * 0.05
    cut_off_95 = np.max(runs) * 0.95

    runs_above_per_95 = np.where(runs > cut_off_95)
    runs_below_per_5 = np.where(runs < cut_off_5)
    print(f'Number of runs below 0.05 cut-off: {runs_below_per_5[0].size}')
    print(f'Number of runs above 0.95 cut-off: {runs_above_per_95[0].size}')

    per_of_runs_out_cut_offs = (runs_above_per_95[0].size + runs_below_per_5[0].size)*runs[0].size/100
    print(f'Percentage of runs out the cut-offs: {per_of_runs_out_cut_offs} \n')

    print(f'Maximum value: {np.max(runs)}')
    print(f'Minimum value: {np.min(runs)}')
    print(f'Mean: {np.mean(runs)}')
    print(f'Mode: {st.mode(runs,axis=0,keepdims=False)}')
    print(f'Median: {np.median(runs)} \n')

    plt.hist(runs,bins=15)
    plt.title('Runs test results')
    plt.ylabel('Frequencies')
    plt.xlabel('Number of runs')
    plt.axvline(x=cut_off_5)
    plt.axvline(x=cut_off_95)
    plt.grid(True)
    plt.show()
    
# Apply the runs test to lossy trace
runs_test(lossy_trace)

# -------- Helper functions --------

# This function counts the appearances of a given segment in a trace
def count_appearances(subtrc,spltd_trc):
    count = 0
    for element in spltd_trc:
        if np.array_equal(subtrc,element):
            count += 1
    return count

# Converts dict values to a list
def values_to_list(dict):
    l = np.array([],dtype=int)
    for v in dict.values():
        l = np.append(l,int(v))
    return l

# Returns the indexes at which a given element appears
def get_indexes(trc,element):
    indexes = np.array([],dtype=int)
    i = 0
    while i < len(trc):
        if np.array_equal(trc[i],element):
            indexes = np.append(indexes,i)
        i += 1
    return indexes

# Counts the appareances of a segment followed by a given element
def count_appearances_followed_by(num,trc_spltd,permutation):
    count = 0
    indxs = get_indexes(trc_spltd,permutation)
    # Vamos recorriendo el arreglo de índices y revisando el primer número del elemento siguiente
    i = 0
    while i < indxs.size and indxs[i] < len(trc_spltd) - 1:
        index = indxs[i]
        if trc_spltd[index+1][0] == num:
            count += 1
        i += 1
    return count

# --------------------------------

"""
The conditional entropy is an indication of the randomness of the next element of a trace, given the past history.
It is calculated by the sum of the frequency of the different possible segment appearances given an order for the Markov Chain states.
With the conditional entropy for different orders, we can choose the order with minimum entropy and acceptable complexity level for the DTMC.

* A conditional entropy value close to 1 means that the next element in the trace is less predictable from its previous states
"""
def get_conditional_entropy(trc, order):
    n_partitions = math.floor((trc.size)/order)
    # Partition the trace to be able to carry out the necessary searches
    trc_splitted = np.array_split(trc,n_partitions)
    # Calculate all the possible permutations of length equal to the order
    perm = list(product([0,1], repeat=order))
    # Count the appearances
    appearances = {}
    
    for element in perm:
        count = count_appearances(element,trc_splitted)
        appearances[element] = count
    
    num_of_ap_fllwd_by_0 = {}
    num_of_ap_fllwd_by_1 = {}

    sum_0_arr = np.array([])
    sum_1_arr = np.array([])

    # Count the appearances of each permutation in the trace followed by 0 and by 1
    for k,v in appearances.items():
        if v != 0:
            num_of_ap_fllwd_by_0[k] = count_appearances_followed_by(0,trc_splitted,k)
            num_of_ap_fllwd_by_1[k] = count_appearances_followed_by(1,trc_splitted,k)

            # Calculates the factors of each sum of the formula
            first_factor_fllwd_by_0 = num_of_ap_fllwd_by_0[k]/n_partitions
            second_factor_fllwd_by_0 = np.log2(num_of_ap_fllwd_by_0[k]/v) if num_of_ap_fllwd_by_0[k] != 0 else 0
            sum_0_arr = np.append(sum_0_arr, (first_factor_fllwd_by_0*second_factor_fllwd_by_0))


            first_factor_fllwd_by_1 = num_of_ap_fllwd_by_1[k]/n_partitions
            second_factor_fllwd_by_1 = np.log2(num_of_ap_fllwd_by_1[k]/v) if num_of_ap_fllwd_by_1[k] != 0 else 0
            sum_1_arr = np.append(sum_1_arr,(first_factor_fllwd_by_1*second_factor_fllwd_by_1))
    
    sum_0 = np.sum(sum_0_arr)
    sum_1 = np.sum(sum_1_arr)

    return -(sum_0+sum_1)



# Gets the conditional entropy of the lossy trace for each order (from 1 to 10)
entropies = {}
order = 1
while order < 11:
    entropies[order] = get_conditional_entropy(lossy_trace,order)
    order += 1

print('-------- ENTROPY AND DTMC INFORMATION -------- \n')
print(f'Entropies: {entropies}')

plt.plot(entropies.values(),entropies.keys(),'bo')
plt.title('Conditional entropy per order')
plt.xlabel('Entropy')
plt.ylabel('Order')
plt.grid(True)
plt.xlim(-0.1,1.1)
plt.show()

dtmc_total = {}

# Calculates the probability tables for the DTMC through frecuency counting
def get_dtmc(trc, order):
    dtmc = []
    pbb = {}
    pbb_fllwd_by_0 = {}
    pbb_fllwd_by_1 = {}
    n_partitions = math.floor((trc.size)/order)
    trc_splitted = np.array_split(trc,n_partitions)
    perm = list(product([0,1], repeat=order))

    for element in perm:
        count = count_appearances(element,trc_splitted)
        if count > 0:
            pbb[element] = count/n_partitions
            pbb_fllwd_by_0[element] = count_appearances_followed_by(0,trc_splitted,element)/count
            pbb_fllwd_by_1[element] = count_appearances_followed_by(1,trc_splitted,element)/count
            probabilities = [element, pbb[element], pbb_fllwd_by_0[element], pbb_fllwd_by_1[element]]
            dtmc.append(probabilities)
    dtmc_total[order] = dtmc

# Gets the DTMC probabilities of the lossy trace for each order (from 1 to 10)
order = 1
while order < 11:
    get_dtmc(lossy_trace,order)
    order += 1

# Writes the DTMC tables to a file
with open('dtmc_tables.txt','w') as tables:
    for k,v in dtmc_total.items():
        col_names = ['State', 'P(i)', 'P(0|i)', 'P(1|i)']
        table = f'Order: {k} \n {tabulate(v,headers=col_names,tablefmt="grid")}\n'
        tables.write(table)