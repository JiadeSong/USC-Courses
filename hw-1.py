import time # declare all your imports in this section
import numpy
import csv


"""
 In this homework, we will be studying two applied topics:
 (1) how to write practical python code!
 (2) how to run meaningful timed experiments by 'simulating' architectures like mapreduce

 This homework is best done after re-starting your machine, and making  sure minimal 'stuff' is open. You may
 need to consult a browser occasionally, as well as the command line, but otherwise please try to minimize
 the programs running  in the background. MAX SCORE POSSIBLE: 100, not including 20 extra credit points.
"""

# Let us begin by giving you a crash course in a simple way to measure timing in python. Timing is actually
# a complicated issue in programming, and will vary depending on  your own system. So to make it uniform
# for all students (in procedure, if not in answer), we are telling  you how you should be measuring timing.

# Before getting started, it's good to install PyCharm, which is an integrated development environment (IDE)
# for writing Python programs and setting up Python projects.

# Uncomment the code fragment below, and run it (hint: on  mac, you can select the lines and use command+/).
# Otherwise, look under  the 'Code' menu.

# start = time.time()
# test_list = list()
# for i in range(0,20000000):
#     test_list.append(i)
# end = time.time()
# print(end - start)
# print len(test_list)
# test_list = None # free up memory

"""
PART I: 15 POINTS
[+5 extra credit points]

Q1. Did the code fragment run without errors? What's the answer? [2+2 points]
ANS: 
    Yes, no errors. 
    Answer is: 
        5.00679016113e-06
        10

Q2: What's the answer when you change range(0,x) from x=10 to x=1000? What  about x= 100000? [3+3 points]
(I don't care about the 'units' of the timing, as long as you're consistent. However, you should look up
the Python doc and try to be precise about what units the code is outputting i.e. is is milliseconds, seconds, etc.?)
ANS:
    For 1000: 
        0.0001060962677
        1000
    For 100000: 
        0.0124819278717
        100000
    The units are in seconds. 

Q3: Given that we 'technically' increased the input by 100x (from x=10 to x=1000) followed by another 100x increase, do
you see a similar 'slowdown' in timing? What are the slowdown factors for both of
the 100x input increases above COMPARED to the original input (when x=10)? (For example, if I went from 2 seconds to
30 seconds when  changing from x=10 to x=1000, then the slowdown  factor is 30/2=15 compared to the original input) [5 points]
ANS:
    0.0001060962677/(5.00679016113e-06) = 21.19 (from x=10 to x=1000)
    0.0124819278717/0.0001060962677 = 117.65 (from x=1000 to x=100000)
    The slowdown in timing is not similar as the x100 of the original factor. For the first multiplying (from x=10 to x=1000), 
    the time was 21.19x increase. However, for the second multiplying (from x=1000 to x=100000), the time was 117.65 increase. 

Note 1: If you have a Mac or Unix-based system, a command  like 'top' should work on the command line.
This is a useful command because it gives you a snapshot of threads/processes running on your machine
and will tell you how much memory is free or bei ng used. You will need to run this command or its equivalent (if 'top' is not applicable to your specific OS) if  you want the extra credit in some of the sections (not necessary if you don't)

Extra Credit Q1 [5 points]: If you run  top right now, how much memory is free on your machine? What is  the value of x
in range(0,x) for which roughly half of your current free memory gets depleted?

ANS:
    3000M
    20000000

Note 2: Please re-comment  the code fragment above after you're done with Q1-3. You should be able
to use/write similar timing code to measure timing for the sections below.

Now we are ready to start engaging with numpy. Make sure you have installed it using either pip or within
Pycharm. We will begin with a simple experiment based on sampling. First, let's review
a list of distributions you can sample from here: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html

Uncomment lines 74 and 78 below.
"""

# let's try sampling values from normal distribution:
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html#numpy.random.normal

# numpy.random.seed(1);
# print('sample = ')
# sample = numpy.random.normal() # The first
# #two parameters of the normal function, namely loc and scale have default values specified. Make sure you understand
# #this, since it will help you read documentation and learn how to use python libraries.
#
# print sample

"""
PART II: 25 POINTS
[+5 extra credit]
Q1. What is the value printed when you run the code above? [3 points]
ANS:
    [ 0.22735867]
    
Q2. Run the code five times. Do you get the same value each time? [3 points]
ANS:
    No.
    
Extra Credit Q2 [5 points]: How do I guarantee that I get the same value each time I run the code above (briefly state
your answer below)? Can you add
a command before sample = numpy.random.normal(size=1) such that every time we execute this code we get the same
'sample' value printed? Write such a command above the 'sample = ' command and comment it out.

ANS:
    Add a seed before the code of getting sample, which is "numpy.random.seed(1);". I add the command "print('sample = ')". 

Q3: What happens if I run the code with no arguments (i.e. delete size=1)? Does the program crash? Briefly
explain the difference compared to the size=1 case. [9 points]
ANS:
    The square brackets will disappear and more digits for the sample value will appear. 
    The program will not crash. 
    
Q4: Add a code fragment below this quoted segment to sample and print 10 values from a laplace distribution
where the peak of the distribution is at position 2.0 and the exponential decay is 0.8. Run the code and paste the
output here. Then comment the code. (Hint: I provided a link to available numpy distributions earlier in this exercise.
Also do not forget to carefully READ the documentation where applicable, it does not take much time) [10 points]
 ANS:
    10 values from the Laplace sample are: 
    [ 1.89041796 -0.36748226  2.0836886   1.88918303  1.86121756  1.66839892
    1.28534914  2.21801614  1.5904181   1.49759475]
    
"""

#paste your response to Q4 below this line.

# print('10 values from the Laplace sample are: ');
# numpy.random.seed(2);
# sample2 = numpy.random.laplace(loc=2.0, scale=0.8, size=10)
# print sample2

# This code section is to print the Laplace distribution with a fixed value of seed=2, which will keep sample values the
# same every time. The distribution position is 2.0 and exponential decay is 0.8 with 10 values.

"""
PART III: 60 POINTS
[+10 extra credit]

This is where we start getting into the weeds with 'simulated' mapreduce. As a first step, please make sure to download
words_alpha.txt. It is a list of words. We will do wordcount, but with a twist: we will 'control' for the distribution
of the words to see how mapreduce performs.
"""

# we will first use the numpy.choice function that allows us to randomly choose items from a list of items:
# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.choice.html#numpy.random.choice

"""
Q1: Read in the words in our vocabulary or dictionary file 'words_alpha.txt' into a list (Hint: do not underestimate this 'simple' task, even
though it is only a few commands; being up close and personal with I/O in any language the first time
can take a couple of trials) Fill in your code in read_in_dictionary_to_list below. Next, right after the function,
 uncomment the print statement and replace the path to words_alpha.txt with wherever it is in your machine. How many words in the list? [10 points].
 ANS:
    370103 words in the list. 
    
Q2: Now we will write a function to 'generate' a document (in generate_document). It takes a list_of_words (e.g., the output
of read_in_dictionary...) as input and outputs another list of length document_length, which you can basically think about
as a 'document' with document_length (possibly repeated) words.
We can use any probability distribution (over the words in the original input list) to generate such a document. Let's try the
choice function as an obvious choice (no pun intended). Go to the link I provided above to read about it. Now, in
generate_document, write the code to generate a document of LENGTH 20 assuming UNIFORM distribution WITH REPLACEMENT over the original
input list (that you read in from file). (Hint: the tricky part in such generators is setting the value for the 'p' argument, but do you
really need to for this question? See the documentation of choice carefully) [10 points]

Extra Credit Q3 [5 points]: Write an alternate generator function (called generate_document_1) that is the same as above
but the distribution is no longer uniform. Instead, assume that the probability of sampling a word with k characters is
inversely proportional to k (Hint: this will mean creating your own p distribution; make sure that it is a valid probability distribution!)

Q3: Now we will run a simple map-reduce function. For context, see the 'master' function. I have already written the
map and reduce functions to make life easier for you. Your primary task will be to run timed experiments. As a first step,
go to the 'master' function and fill in the rest of the code as required. At this point you should be able to run
an experiment (by calling master) and output a file with word counts. [15 points]


Q4: Use the code in the first part of this exercise on 'timing' to plot various curves (you can use Excel or another tool to
plot the curves and only use the code here for collecting 'raw' data) where the length of the document is on the x axis
and varies from 10-100000 in increments of 10,000, and the y-axis is time units. Note that this means you will be computing data for approximately 100000/10000 = 10 x,y points. In practice, we would be considering more points to get a smoother curve, but it is unnecessary for this assignment.

<Note: Write additional functions as you see fit to collect/write out the raw data.
Do not modify master for this question; we will use it for grading Q3. For each length (i.e. each x value on your plot) in
 your experiment, you MUST generate 10 documents (to account for randomness of sampling), and then average the data over the ten. In the plots below, make sure to show error bars.>

 Specifically:
  (a) Plot a curve showing how much time the mapper takes as the length of a document increases.
   (b) Plot a curve showing how much time the reducer takes as the length of a document increases
   (hint: review the start-end timing methodology at the beginning of this file, and record 'time' variables where
   appropriate to help you get the data for the plot)
  [25 points]

    Ans: See the def master2 below. 

  EXTRA CREDIT [5 points]: Use 'top' to collect data on memory consumption and give me a single plot where memory
  consumption (in standard units of your choice) is on the y-axis instead of time. Is memory consumption linear
  in the length of the documents? If the program terminates quickly for small-length documents, you can plot the curve
  starting from a 'higher' length like 20,000
"""

def read_in_dictionary_to_list(input_file):
    with open(input_file) as f:
        contents_list = f.readlines()
        contents_list1 = [x.replace("\r\n", "") for x in contents_list]
        # Remove the \r\n in the list reading from the test file.
    return contents_list1
    # this function must return a list. Normally, good software engineering requires you to deal with 'edge cases' such
    # as an empty file, or a file that does not exist, but you can safely assume the input_file will be the path
    # to words_alpha.txt.
    #pass is a placeholder that does nothing at all. It is a syntactic device; remove and replace it with your own code when ready

print len(read_in_dictionary_to_list(input_file="/Users/jiadesong/Desktop/ISE540/hw1-JiadeSong/words_alpha.txt"))
# This command is for naming the list
# what this hopefully also conveys is that you actually have to
# call a function with inputs to get it to 'do something'. Otherwise, it's just defined and sitting there...waiting.

def generate_document(list_of_words, document_length):
    gendoc = numpy.random.choice(list_of_words, document_length, replace=True)
    return gendoc

def generate_document_1(list_of_words, document_length):
    num_list = []
    local_count = 0
    for n in list_of_words:
        kk = 1/float(len(n))
        num_list.append(kk)
        local_count += kk
    # print local_count
    prob_list = []
    for m in num_list:
        # print float(m)/local_count
        prob_list.append(float(m)/float(local_count))

    gendoc = numpy.random.choice(list_of_words, document_length, replace=True, p=prob_list)
    return gendoc

# print generate_document_1(read_in_dictionary_to_list(input_file="/Users/jiadesong/Desktop/ISE540/words_alpha.txt"),20)

# this function is only for instructional purposes, to help you understand dicts. Don't forget to call it to get it to do something!
def playing_with_dicts():
    test_dict = dict()
    # keys and values can be any data type. In this case, the key is string and the value is integer. Values can potentially be
    # complicated data structures, even Python dictionaries themselves!
    sample_words = ['the', 'cow', 'jumped', 'over','the','moon']
    for s in sample_words:
        if s not in test_dict: # is s a key already in test_dict?
            test_dict[s] = 0 # initialize count of s to 0
        test_dict[s] += 1 # at this point, s HAS to be a key in test_dict, so we just increment the count
    print test_dict # what happens? make sure to play and experiment so that you get the hang of working with dicts!

#playing_with_dicts() # uncomment for the function above to do something.


"""
Background on global_data (if you don't want to read this, make sure to check out the code fragment at the very end
of this file; it provides some intuition on what I've explained below and gives you a way to try things out!):

Just like in class, the mapper takes a 'document' i.e. a list that you generated
as input. The tricky part is the output, since in an 'implemented' mapreduce system, there is a mechanism to ensure that the 'key-value' pairs that are
emitted by the mapper are routed correctly to the reducer (as we discussed in class, all that this means is that the key-value
pairs that have the same key are guaranteed to be routed to the same reducer. Recall that reducers and mappers do not share information
otherwise, thus being embarrassingly parallel). In this implementation, because everything is being done within a single
program we will use a global data structure called 'global_data' to 'simulate' this mechanism. As the code shows, global_data
is a 'dict()' which is a python data structure (DS) that supports keys and values. See my code fragment playing_with_dicts, play with
them! They're the most 'pythonic' DS of all.

So where does global_data come in? We use it to store all key-value pairs emitted by the mapper. To do so, the 'value'
in global_data is actually a list. The list contains all the values emitted by individual mappers. For example, imagine
that the word 'the' occurs thrice in one document and twice in another document. global_data now contains a key 'the'
with value [3,2]. The reducer will independently receive key-value pairs from global_data.

Reduced_data works in a similar way; it records the outputs of the reducer. You will have to write out the outputs
of reduced_data to file.

"""

global_data = dict()
reduced_data = dict()

# already written for you
# as a unit test, try to send in a list (especially with repeated words) and then invoke print_dict using global_data as input
def map(input_list):
    local_counts = dict()
    for i in input_list:
        if i not in local_counts:
            local_counts[i] = 0
        local_counts[i] += 1
    for k, v in local_counts.items():
        if k not in global_data:
            global_data[k] = list()
        global_data[k].append(v)

def print_dict(dictionary): # helper function that will print dictionary in a nice way
    for k, v in dictionary.items():
        print k, ' ', v


#already written for you
def reduce(map_key, map_value): # remember, in python we don't define or fix data types, so the 'types' of these arguments
    #can be anything you want!
    total = 0
    if map_value:
        for m in map_value:
            total += m

    reduced_data[map_key] = total


def master(input_file, output_file): # see Q3
    word_list = read_in_dictionary_to_list(input_file)

    # write the code below (replace pass) to generate 10 documents, each with the properties in Q2. You can use a
    # list-of-lists i.e. each document is a list, and you could place all 10 documents in an 'outer' list. [6 points]
    outer = []
    for m in range(10):
        outer.append(generate_document(word_list, 20))

    # Call map over each of your documents (hence, you will have to call map ten times, once over each of your documents.
    # Needless to say, it's good to use a 'for' loop or some other iterator to do this in just a couple of lines of code [ 6 points]
    for k in outer:
        map(k)

    for k, v in global_data.items(): # at this point global_data has been populated by map. We will iterate over keys and values
        # and call reduce over each key-value pair. Reduce will populate reduced_data
        reduce(k, v)

    # write out reduced_data to output_file. Make sure it's a comma delimited csv, I've provided a sample output in sample_output.csv [3 points]
    with open(output_file, 'w') as f:
        for key, value in reduced_data.items():
            f.write('%s,%s\n' % (key, value))
    # print reduced_data


master(input_file="/Users/jiadesong/Desktop/ISE540/words_alpha.txt", output_file='/Users/jiadesong/Desktop/ISE540/output.csv') # uncomment this to invoke master

# simple test for map/reduce (uncomment and try for yourself!)
# map(['the', 'cow', 'jumped', 'over','the','moon'])
# map(['the', 'sheep', 'jumped', 'over','the','sun'])
# print global_data
# for k, v in global_data.items():
#     reduce(k,v)
# print reduced_data


'''

def master2(input_file, output_file, doclen): # see Q3
    word_list = read_in_dictionary_to_list(input_file)
    totaltime=0
    for rr in range(10):
        # write the code below (replace pass) to generate 10 documents, each with the properties in Q2. You can use a
        # list-of-lists i.e. each document is a list, and you could place all 10 documents in an 'outer' list. [6 points]
        outer = []
        for m in range(10):
            outer.append(generate_document(word_list, doclen))

        # Call map over each of your documents (hence, you will have to call map ten times, once over each of your documents.
        # Needless to say, it's good to use a 'for' loop or some other iterator to do this in just a couple of lines of code [ 6 points]

        # start = time.time()
        for k in outer:
            map(k)
        # end = time.time()
        # totaltime += end - start

        start = time.time()
        for k, v in global_data.items(): # at this point global_data has been populated by map. We will iterate over keys and values
            # and call reduce over each key-value pair. Reduce will populate reduced_data
            reduce(k, v)
        end = time.time()
        totaltime += end - start

        # write out reduced_data to output_file. Make sure it's a comma delimited csv, I've provided a sample output in sample_output.csv [3 points]
        with open(output_file, 'w') as f:
            for key, value in reduced_data.items():
                f.write('%s,%s\n' % (key, value))
        # print reduced_data
    print(totaltime/10)
    return totaltime

for k in [10,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]:
    tt = []
    # uncomment this to invoke master
    tt.append(master2(input_file="/Users/jiadesong/Desktop/ISE540/words_alpha.txt", output_file='/Users/jiadesong/Desktop/ISE540/output.csv', doclen = k)/10)

'''



