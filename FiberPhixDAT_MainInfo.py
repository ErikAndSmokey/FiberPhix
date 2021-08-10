import tdt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



entry_df= pd.ExcelFile('YOUR PATH TO USER ENTRY SPREADSHEET').parse()
path_to_file = str(entry_df['Path to tank file'][0])

class Main_Info():

    def __init__ (self,path_to_file):
        self.path = path_to_file


    def get_main_info(self):
        self.data = tdt.read_block(self.path)
        self.streams = self.data.streams
        self.epocs = self.data.epocs
        self.blockname = self.data.info.blockname
        
        
information = Main_Info(path_to_file)
information.get_main_info()

stream_id_df = pd.DataFrame(information.streams.keys(), columns = ['Stream IDs Available'])
epoc_id_df = pd.DataFrame(information.epocs.keys(), columns = ['Epoc IDs Available'])

with pd.ExcelWriter(f'PATH TO THE FOLDER TO STORE IDS') as writer:
    stream_id_df.to_excel(writer, sheet_name = 'Stream IDs')
    epoc_id_df.to_excel(writer, sheet_name = 'Epoc IDs')
