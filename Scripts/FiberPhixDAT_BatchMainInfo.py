import tdt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


#Import the user entry spreadsheet (i.e. 'User Values') for use to create variables
entry_df= pd.ExcelFile('C:\\PATH TO USER VALUES SS\\User Values' + '.xlsx').parse()


#Pull variable names from the user entry spreadsheet
path_to_file = str(entry_df['Path to tank file'][2])


#Class for grabbing the stream Id's and epoc Id's
class Main_Info():

    def __init__ (self,path_to_file):
        self.path = path_to_file


    def get_main_info(self):
        self.data = tdt.read_block(self.path)
        self.streams = self.data.streams
        self.epocs = self.data.epocs
        self.blockname = self.data.info.blockname
        self.session_length = self.data.info.duration.seconds

class Get_Epocs():
    
    def __init__ (self, epoc_info):
        self.epoc_info = epoc_info
        self.epoc_names = list(self.epoc_info.keys())
        
    def all_epoc_info(self):
        self.info_by_epoc = []
        for i in self.epoc_names:
            self.info_by_epoc.append(self.epoc_info[i])
        self.df_builder = dict(zip(self.epoc_names,self.info_by_epoc))



for i in os.listdir(path_to_file):
    information = Main_Info(path_to_file + '/' + i)
    information.get_main_info()

    epocs = Get_Epocs(information.epocs)
    epocs.all_epoc_info()

    stream_id_df = pd.DataFrame([[list(information.streams.keys()),information.session_length]], columns = [['Stream IDs Available', 'Session Length (Seconds)']])
    epoc_df = pd.DataFrame(epocs.df_builder)

    with pd.ExcelWriter(f'PATH TO INFO STORAGE FOLDER\\Information from {information.blockname}' + '.xlsx') as writer:
        stream_id_df.to_excel(writer, sheet_name = 'Stream IDs')
        for i in epocs.epoc_names:
            df = pd.DataFrame(list(zip(epocs.epoc_info[i]['data'], epocs.epoc_info[i]['onset'], epocs.epoc_info[i]['offset'])),
                              columns = ['Port', 'Onsets', 'Offsets'])
            df.to_excel(writer, sheet_name = f'EPOC {i}')