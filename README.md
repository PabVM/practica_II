# practica_II
The purpose of this project is to analyze the lossy trace of a set of experiments over the Sigfox network and determine its behavior regarding lost messages.

The following steps are described for the execution of the experiments:

1. Load the /lopy/main.py script to the LoPy. This script sends a message every 20 seconds through the Sigfox Network, up to total of 20.000 messages.
2. Connect the LoPy device to the current, so that it can send the messages without interruptions.
3. Execute the script /requests/get_msgs_from_sigfox.py. This script consults every 2000 seconds the last 100 messages received by the Sigfox network and adds them to the data set 'experiments.json'.
4. Once the results of the experiments are obtained, run the script /analysis/mta.py as follows:

`python3 mta.py experiments.json > results.txt`

This will allow to record the results of the analysis of the results through the application of the MTA algorithm, described by Konrad et al. in the research "A Markov-Based Channel Model Algorithm for Wireless Networks".

The execution of this script results in the following graphs, plus a summary of statistical data and probability tables for modeling the loss traces as Discrete-Time Markov Chains (DTMC):

- Lengths of the loss segments in the trace of the messages
- Variation of the standard deviations of the loss lengths according to the size of the sample
- Lengths of the error-free segments in the trace of the messages
- Results of the Runs Test application to prove the stationarity of the loss trace
- The conditional entropies vs. the order of the DTMC