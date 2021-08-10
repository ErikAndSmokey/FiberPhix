import tdt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import signal
from scipy import stats


entry_df= pd.ExcelFile('PATH TO USER ENTRY SPREADSHEET').parse()
path_to_file = str(entry_df['Path to tank file'][0])
exp_ch_name = str(entry_df['Signal Channel ID'][0])
control_ch_name = str(entry_df['Control Channel ID'][0])
time_pre_epoc = int(entry_df['Time to grab before each epoc (seconds)'][0])
time_post_epoc = int(entry_df['Time to grab after each epoc (seconds)'][0])
epoc_name = str(entry_df['Epoc ID'][0])
eoi_num = int(entry_df['Epoc Type to grab?']) #The epoc type of interest... for centering around that type of epoc later

class All_Epocs():

    def __init__(self, path_to_file,epoc_name,eoi_num):
        self.path = path_to_file
        self.name = epoc_name
        self.data = tdt.read_block(self.path)
        self.eoi = eoi_num

    def make_epoc_df(self):
        self.all_epocs_df_builder = {'Epoc Data':self.data.epocs[self.name]['data'],
                                    'Onsets':self.data.epocs[self.name]['onset'],
                                    'Offsets':self.data.epocs[self.name]['offset']}
        self.all_epocs_df = pd.DataFrame(self.all_epocs_df_builder)
        self.types_of_epocs = [int(i) for i in list(self.all_epocs_df['Epoc Data'].unique())]
        self.values_epoc_dict = dict(zip(self.types_of_epocs,list(range(1,(len(self.types_of_epocs)+1)))))
        self.all_epocs_df['Epoc ID Number'] = self.all_epocs_df['Epoc Data'].map(self.values_epoc_dict)
        
    def isolate_epoc_type(self):
        self.eoi_df = self.all_epocs_df[self.all_epocs_df['Epoc ID Number'] == self.eoi]
    
class EOI_Tools():
    
    def __init__(self,path_to_file, exp_ch, control_ch, pre_epoc_time, post_epoc_time, eoi_df):
        self.exp = exp_ch
        self.control = control_ch
        self.pre = pre_epoc_time
        self.post = post_epoc_time
        self.path = path_to_file
        self.eoi_df = eoi_df
        self.onsets = list(self.eoi_df['Onsets'])
        
        
    def data_by_epoc(self):
        self.lengths = []
        for i in self.onsets:
            reader = tdt.read_block(self.path, t1 = (i-self.pre), t2 = (i+(self.post + 0.1)))
            gather_times = (np.linspace(1,len(reader.streams[self.exp]['data']),len(reader.streams[self.exp]['data']))/reader.streams[self.exp].fs)
            self.lengths.append(gather_times)
        self.min_time = min(len(i) for i in self.lengths)
        self.times = self.lengths[0][:self.min_time]
        self.data = []
        for i in self.onsets:
            reader = tdt.read_block(self.path, t1 = (i-self.pre), t2 = (i+(self.post + 0.1)))
            temp_exp_data = reader.streams[self.exp]['data']
            temp_time = (np.linspace(1,len(reader.streams[self.exp]['data']),
                                     len(reader.streams[self.exp]['data']))/reader.streams[self.exp].fs)
            temp_exp_detrend = signal.detrend(temp_exp_data)
            temp_exp_coefs = np.polyfit(temp_exp_detrend,temp_exp_data,1)
            temp_exp_fit = [temp_exp_coefs[0]*i+temp_exp_coefs[1] for i in temp_exp_detrend]
            temp_control_data = reader.streams[self.control]['data']
            temp_control_data_detrend = signal.detrend(temp_control_data)
            temp_control_coefs = np.polyfit(temp_control_data_detrend,temp_control_data,1)
            temp_control_fit = [temp_control_coefs[0]*i+temp_control_coefs[1] for i in temp_control_data]
            
            temp_epoc_df_builder = {'Time': temp_time,
                                   'Exp Signal': temp_exp_fit,
                                   'Control Signal': temp_control_fit}
            temp_epoc_df = pd.DataFrame(temp_epoc_df_builder)
            
            temp_epoc_scale_coefs = np.polyfit(temp_epoc_df['Control Signal'], temp_epoc_df['Exp Signal'],1)
            temp_epoc_df['Scaled Control'] = [temp_epoc_scale_coefs[0]*i+temp_epoc_scale_coefs[1] for i in temp_epoc_df['Control Signal']]
            temp_epoc_df['Signal - Control'] = temp_epoc_df['Exp Signal'] - temp_epoc_df['Scaled Control']
            temp_epoc_df['F/F'] = temp_epoc_df['Signal - Control']/temp_epoc_df['Scaled Control']
            temp_epoc_df['Normalized F/F (Z-Score)'] = stats.zscore(temp_epoc_df['F/F'])
            self.data.append(list(temp_epoc_df['Normalized F/F (Z-Score)'][:self.min_time]))
        non_zero_index = np.linspace(1,len(self.data), len(self.data), dtype=int)
        self.eoi_data_df = pd.DataFrame(self.data, columns = self.times, index = non_zero_index)
        
    def heatmapper(self):
        num_secs = (self.pre + self.post + 1)
        ticks = list(self.eoi_data_df.columns)
        num_columns = len(self.eoi_data_df.columns)
        xticks = np.linspace(0,num_columns-1,num_secs, dtype = int)
        xticklabels = [int(ticks[i]) for i in xticks]
        fig,axes = plt.subplots(figsize = (24,12))
        sns.heatmap(self.eoi_data_df, cmap = 'viridis')
        plt.xticks(ticks= xticks,labels = xticklabels)
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Trial (# w/n Session)')
        plt.tight_layout()
    
            
        

shock_epocs = All_Epocs(path_to_file, epoc_name, eoi_num)
shock_epocs.make_epoc_df()
shock_epocs.isolate_epoc_type()
shock_epocs.eoi_df


analyze_shocks= EOI_Tools(path_to_file, exp_ch_name, control_ch_name, time_pre_epoc, time_post_epoc, shock_epocs.eoi_df)
analyze_shocks.data_by_epoc()
analyze_shocks.heatmapper()
plt.savefig('PATH TO FOLDER FOR FIGURE STORAGE', dpi = 600)