import csv
import os
import pickle
import string
import sys

import numpy as np
import pandas as pd

from itertools import cycle, compress
from scipy.sparse import csr_matrix

all_files = os.listdir("./ADM-2019-Assignment-1-data-T-SF-1/")

class read_input:
    
    '''
    This class asks the user if they want to encode or decode, what the filename is and what the datatype is. 
    If a non-existing answer is given to any of the questions, the input to that question is asked again. 
    For the first question only 'en' and 'de' are valid answers. The second question takes any filename in the data folder
    as a correct answer, and the third question takes the datatype as given in the filename as a correct answer. 
    
    Once all of the input is legitimate input, this class reads the file.
    '''
    
    def __init__(self):
        
        self.decode_extensions = ['rle','bin','dic','for','dif'] #All possible extensions for encoded data
        self.all_files_to_encode = os.listdir("./ADM-2019-Assignment-1-data-T-SF-1/") #All possible files to encode
        self.all_files_to_decode = [x for x in os.listdir() if x.endswith(tuple(self.decode_extensions))] #All possible files to decode
        
    def get_input(self, which = 'all'):
        
        #Asks the user for input
        
        if (which == 'codetype') | (which == 'all'):
            self.codetype_input = input("Decode or Encode? (de/en)")
        if (which == 'filename') | (which =='all'):
            self.filename_input = input("Filename:")
        if (which == 'datatype') | (which == 'all'):
            self.datatype_input = input("Datatype:")

        
    def check_codetype(self):
        
        #Check if given codetype is valid
        
        return self.codetype_input in ['en','de']
        
    def check_filename(self):
        
        #Check if given filename is valid
        
        if self.codetype_input == 'en': #Option chosen is to encode
            return self.filename_input in self.all_files_to_encode
        else: #Option chosen is to decode
            return self.filename_input in self.all_files_to_decode

    def check_datatype(self):
        
        #Check if datatype in filename is the same as given datatype by user
        
        datatype_from_filename = self.filename_input.split('.')[0].split('-')[-1]
        
        return datatype_from_filename == self.datatype_input
            
    def read_data(self):
        
        #Reads the data
        
        print('Reading data...')
        data_input = []
        
        if self.codetype_input == 'en': #Option chosen is to encode
            with open('./ADM-2019-Assignment-1-data-T-SF-1/'+self.filename_input, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter='\n') #Read the file to be encoded
                for row in reader: 
                    data_input.append(row[0])
                data_input= np.array(data_input, dtype=self.datatype_input)
        else:  #Option chosen is to decode
            with open(self.filename_input, "rb") as f:
                data_input = pickle.load(f) #Read the file to be decoded
                f.close()
            
        return data_input

    def run_it(self):
        
        #This function runs everything together. It first makes sure the user input is correct, then it reads the data.
        
        self.get_input()
        while not self.check_codetype() & self.check_filename() & self.check_datatype():
            if not self.check_codetype():
                print('Give correct type: (de/en).')
                self.get_input(which = 'codetype')
            elif not self.check_filename():
                print('Give existing filename.')
                self.get_input(which = 'filename')
            elif not self.check_datatype():
                print('Give correct datatype.')
                self.get_input(which = 'datatype')
        if self.datatype_input == 'string':
            self.datatype_input == 'str'

        return self.read_data()


class encode:
    
    '''
    This class encodes data. It takes as input the data to be encoded, the filename and the datatype (int8/string/etc.)
    If the full pipeline is run, it encodes the data in all five of the following ways:
    
    - Binary encoding
    - Dictionary encoding
    - Run-Length encoding
    - Frame of Reference encoding
    - Differential encoding
    
    And then saves each encoding individually with the correct filename and extension.
    '''

    def __init__(self, data, filename, datatype):
        
        self.data = data
        self.filename = filename.split('.')[0]
        self.datatype = datatype
        
    def convert_datatype(self, entry):
        
        '''
        Converts the input (entry) to the datatype as specified with creating this class.
        '''
        
        if self.datatype == 'int8':
            return np.int8(entry)
        elif self.datatype == 'int32':
            return np.int32(entry)
        elif self.datatype == 'int64':
            return np.int64(entry)
        elif self.datatype == 'string':
            return str(entry)
        else:
            return entry
            
    def binary(self):
        #.bin
        #int only
        print('Binary Encoding...')

        encoded = []

        for n in self.data:

            if n!=0:
                b_rev=[]
                while n!=0:
                    #Convert to powers of 2
                    b_rev.append(int(n)%2)
                    n= int(n)/2
                b=b_rev[::-1] #Reverse
                b=int(''.join(str(e) for e in b)) #Put them together
            else:
                b=0
            encoded.append(b)
        
        return np.array(encoded, dtype=self.datatype)
        
    
    def dic(self):
        print('Dictionary Encoding...')
        unique = sorted(np.unique(self.data))
        a=np.arange(0,len(unique))
        
        dic={}
        for key, value in zip(unique, a):
            dic[key] = value
            
        return tuple((dic, np.array(list(map(dic.get, self.data)))))
        
    def rle(self):
        #.rle
        
        #Run-length encoding
        
        print('Run-Length Encoding...')

        encoded = []
        count = 1
        prev = None
        
        for c in self.data: #Loop through each entry in the data
            if c != prev: #If the entry is not the same as the previous one
                if prev != None: #And the entry is not the first entry in the data
                    encoded.append((self.convert_datatype(prev),self.convert_datatype(count))) #Append the entry and its count
                    count = 1 #Start counting at 1 again

            else: #If the entry is the same as the previous one
                count += 1 #Up the count

            prev = c
    
        encoded.append((self.convert_datatype(prev),self.convert_datatype(count))) #Add the last entry with its count
        return np.array(encoded)

    def frame_or(self):
        #.for
        #int only

        print('Frame of Reference Encoding...')

        #data_num = [float(x) for x in self.data] #Make sure the data is number-type
        frame_of_ref = int(round(np.mean(self.data))) #Take mean of data as frame of reference

        encoded_int8 = []
        encoded_rest = []
        encoded_rest.append(frame_of_ref) #First entry is going to be frame of reference

        for i, d in enumerate(self.data): #Loop over data

            diff = d - frame_of_ref #Calculate difference of data entry with frame of ref

            if (-129 < diff) & (diff < 127): #8 bits
                #127 = 1111111
                encoded_int8.append(np.int8(diff))

            else: #Difference doesn't fit in 8 bits
                encoded_int8.append(np.int8(127))
                encoded_rest.append(np.int32(d))

        return (np.array(encoded_int8, dtype='int8'), np.array(encoded_rest, dtype='int32'))

    def dif(self):
        #.dif
        #int only

        print('Differential Encoding...')
        data_float = [float(x) for x in self.data] #Make sure data is in number format
        prev = data_float[0]

        encoded_int8 = []
        encoded_rest = []
        encoded_rest.append(prev)

        for d in data_float[1:]:
            diff = d-prev

            if (-129 < diff) & (diff < 127): #8 bits
                encoded_int8.append(np.int8(diff))
                prev = d

            else: #Difference doesn't fit in 8 bits
                encoded_int8.append(np.int8(127))
                encoded_rest.append(np.int32(d))
                prev=d
                
        return (np.array(encoded_int8, dtype='int8'), np.array(encoded_rest, dtype='int32'))
    
    def encode_all(self):
        
        self.encoded_dict = {}
        
        self.encoded_dict['rle'] = self.rle()
        self.encoded_dict['dic'] = self.dic()
        
        if 'int' in self.datatype:
            self.encoded_dict['bin'] = self.binary()
            self.encoded_dict['for'] = self.frame_or()
            self.encoded_dict['dif'] = self.dif()
        
    def save_file(self, extension):
        
        print('Saving encoded file... {}'.format(self.filename+'.'+extension))
        with open(self.filename + '.'+extension, "wb") as f:
            pickle.dump(self.encoded_dict[extension], f)
        f.close()
        
    def save_all(self):
        
        for k in self.encoded_dict.keys():
            self.save_file(k)
            
    def run_it(self):
        self.encode_all()
        self.save_all()

class decode:

    '''
    This class decodes a datafile that is encoded in one of the following five ways:

    - Binary encoding
    - Dictionary encoding
    - Run-Length encoding
    - Frame of Reference encoding
    - Differential encoding

    It decides the decoding method based on the extension in the filename, then decodes the file.

    '''
    
    def __init__(self, data, filename, datatype):
        self.data = data
        self.filename = filename
        self.datatype = datatype
        
    def convert_datatype(self, entry):
        
        '''
        Converts the input (entry) to the datatype as specified with creating this class.
        '''
        
        if self.datatype == 'int8':
            return np.int8(entry)
        elif self.datatype == 'int32':
            return np.int32(entry)
        elif self.datatype == 'int64':
            return np.int64(entry)
        elif self.datatype == 'str':
            return str(entry)
        else:
            return entry
        
    def binary(self):
        #.bin (binary encoding)
        #int only
        #decoded = np.array([int(x, 2) for x in self.data])
        #return [self.convert_datatype(x) for x in decoded]

        decoded = []
        
        for n in self.data:
            d=0
            n_s=str(n)
            l=len(n_s)-1
            for i in range(len(n_s)):
                d=int(n_s[i])*pow(2,l-i)+d
            
            decoded.append(d)
        
        return decoded

    def dic(self):
        #.dic (dictionary encoding)
        
        res = dict((v,k) for k,v in self.data[0].items()) #Flip dictionary
        decoded = list(map(res.get, self.data[1])) #Use it to decode data
        return [self.convert_datatype(x) for x in decoded] #Return in correct datatype
        
    def rle(self):
        #.rle (run-length encoding)
        #Encoded data contains tuples (original value, run length)
        
        decoded = []

        for pair in self.data:
            decoded.extend([self.convert_datatype(pair[0])]*int(pair[1]))

        return decoded
    
    def frame_or(self):
        
        #.for (Frame of reference encoding)
        #int only
        
        decoded = []
        encoded_rest = self.data[1]
        encoded_int8 = self.data[0]
        frame_of_ref = encoded_rest[0] #The frame of reference value

        encoded_rest=np.delete(encoded_rest, 0)

        for d in encoded_int8: #Loop over each entry

            if d == 127: #if value not encoded
                decoded.append(self.convert_datatype(encoded_rest[0])) #take value from not encoded ones
                encoded_rest=np.delete(encoded_rest, 0)
                
            else: #if value encoded
                num = frame_of_ref + d #otherwise decoded value is encoded value plus frame of reference
                decoded.append(self.convert_datatype(num))

                
        return decoded
            
    def dif(self):
        #.dif (differential encoding)
        #int only
        
        encoded_int8 = self.data[0]
        encoded_rest = self.data[1]
        prev=encoded_rest[0] #store previous value

        decoded=[]
        decoded.append(self.convert_datatype(prev))

        for d in encoded_int8:

            if d == 127: #if value not encoded
                prev=encoded_rest[0]
                decoded.append(prev)
                encoded_rest=np.delete(encoded_rest, 0)

            else:
                enc = d+prev #previous plus current
                prev = enc
                decoded.append(self.convert_datatype(enc))
                
        return decoded
    
    
    def run_it(self):

        '''
        This function takes the file extension and runs the correct decoding function.
        '''
        
        extension = self.filename.split('.')[1]
        print('Decoding the data...')
        if extension == 'rle':
            return self.rle()
        elif extension == 'dic':
            return self.dic()
        elif extension == 'for':
            return self.frame_or()
        elif extension == 'bin':
            return self.binary()
        elif extension == 'dif':
            return self.dif()


if __name__ == "__main__":

    user_input = read_input()
    data = user_input.run_it()

    if user_input.codetype_input == 'en':

        encoding = encode(data, user_input.filename_input, user_input.datatype_input)
        encoding.run_it()

    else: #user_input == 'de'

        decoding = decode(data, user_input.filename_input, user_input.datatype_input)
        decoding.run_it()
