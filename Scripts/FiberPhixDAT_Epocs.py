import tdt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import signal
from scipy import stats



#Import the user entry spreadsheet (i.e. 'User Values') for use to create variables
entry_df= pd.ExcelFile('C:\\PATH TO USER VALUES SS\\User Values.xlsx').parse() #Update with file path to 'User Values' spreadsheet

#Pull variable names from the user entry spreadsheet
path_to_file = str(entry_df['Path to tank file'][0])
exp_ch_name = str(entry_df['Signal Channel ID'][0])
control_ch_name = str(entry_df['Control Channel ID'][0])
time_pre_epoc = int(entry_df['Time to grab before each epoc (seconds)'][0])
time_post_epoc = int(entry_df['Time to grab after each epoc (seconds)'][0])
epoc_name = str(entry_df['Epoc ID'][0])
eoi_port = entry_df['Epoc Port to grab?'][0] #The epoc type of interest... for centering around that type of epoc later
eoi_type = entry_df['Epoc Type?'][0]

#Instantiating this class gathers all the available epocs under a given epoc name, with a separate
#module that allows the user to define a epoc of interest (via number in 'User Values' spreadsheet)
class All_Epocs():

    def __init__(self, path_to_file,epoc_name,eoi_port):
        self.path = path_to_file
        self.name = epoc_name
        self.data = tdt.read_block(self.path)
        self.eoi = eoi_port
        #Makes a DF of all onsets/offsets/types of epocs under a given epoc name
    def make_epoc_df(self):
        self.all_epocs_df_builder = {'Epoc Data':self.data.epocs[self.name]['data'],
                                    'Onsets':self.data.epocs[self.name]['onset'],
                                    'Offsets':self.data.epocs[self.name]['offset']}
        self.all_epocs_df = pd.DataFrame(self.all_epocs_df_builder)
        #Isoate a subset of those epocs by epoc_num <-- taken from 'User Values', use int only...
        #Created list of available epocs in make_epoc_df
    def isolate_epoc_type(self):
        if str(self.eoi).upper() == 'ALL':
            self.eoi_df = self.all_epocs_df
        else:
            self.eoi_df = self.all_epocs_df[self.all_epocs_df['Epoc Data'] == self.eoi]



#After isolating your epoc type of interest    
class EOI_Tools():
    
    def __init__(self,path_to_file, exp_ch, control_ch, pre_epoc_time, post_epoc_time, eoi_df, blockname):
        self.exp = exp_ch
        self.control = control_ch
        self.pre = pre_epoc_time
        self.post = post_epoc_time
        self.path = path_to_file
        self.eoi_df = eoi_df
        self.onsets = list(self.eoi_df['Onsets'])
        self.blockname = blockname
        
   #iterates through all onsets, centering on user defined legnths pre and post epoc (taken from 'User Values').     
    def data_by_epoc(self):
         #Identifies the shortest session length at a given epoc,
        self.lengths = []
        for i in self.onsets:
            reader = tdt.read_block(self.path, t1 = (i-self.pre), t2 = (i+(self.post + 0.1)))
            gather_times = (np.linspace(1,len(reader.streams[self.exp]['data']),len(reader.streams[self.exp]['data']))/reader.streams[self.exp].fs)
            self.lengths.append(gather_times)
        self.min_time = min(len(i) for i in self.lengths)
        self.times = self.lengths[0][:self.min_time]
        self.data = []
    #iterates through all onsets, centering on user defined legnths pre and post epoc (taken from 'User Values'). The following method is similar to that of what is used in FiberPhixDAT
        for i in self.onsets:
            reader = tdt.read_block(self.path, t1 = (i-self.pre), t2 = (i+(self.post + 0.1)))
            temp_exp_data = reader.streams[self.exp]['data']
            temp_time = (np.linspace(1,len(reader.streams[self.exp]['data']),
                                     len(reader.streams[self.exp]['data']))/reader.streams[self.exp].fs)
            #Performs signal detrend on signal channel
            temp_exp_detrend = signal.detrend(temp_exp_data)
            temp_exp_coefs = np.polyfit(temp_exp_detrend,temp_exp_data,1)
            temp_exp_fit = [temp_exp_coefs[0]*i+temp_exp_coefs[1] for i in temp_exp_detrend]
            temp_control_data = reader.streams[self.control]['data']
            #Performs signal detrend on control channel within the same window
            temp_control_data_detrend = signal.detrend(temp_control_data)
            temp_control_coefs = np.polyfit(temp_control_data_detrend,temp_control_data,1)
            temp_control_fit = [temp_control_coefs[0]*i+temp_control_coefs[1] for i in temp_control_data]
            
            temp_epoc_df_builder = {'Time': temp_time,
                                   'Exp Signal': temp_exp_fit,
                                   'Control Signal': temp_control_fit}
            temp_epoc_df = pd.DataFrame(temp_epoc_df_builder)
            
            #Fit control to signal channel and perform calcs for delta f/f
            temp_epoc_scale_coefs = np.polyfit(temp_epoc_df['Control Signal'], temp_epoc_df['Exp Signal'],1)
            temp_epoc_df['Scaled Control'] = [temp_epoc_scale_coefs[0]*i+temp_epoc_scale_coefs[1] for i in temp_epoc_df['Control Signal']]
            temp_epoc_df['Signal - Control'] = temp_epoc_df['Exp Signal'] - temp_epoc_df['Scaled Control']
            temp_epoc_df['F/F'] = temp_epoc_df['Signal - Control']/temp_epoc_df['Scaled Control']
            temp_epoc_df['Normalized F/F (Z-Score)'] = stats.zscore(temp_epoc_df['F/F'])
            self.data.append(list(temp_epoc_df['Normalized F/F (Z-Score)'][:self.min_time]))
        
        #Create a non-zero index for user readability
        non_zero_index = np.linspace(1,len(self.data), len(self.data), dtype=int)

        #Build dataframe of all delta f/f values from epoc of interest
        self.eoi_data_df = pd.DataFrame(self.data, columns = self.times, index = non_zero_index)
    

    #Plots a heatmap of all data from self.eoi_data_df   
    def heatmapper(self):
        num_secs = (self.pre + self.post + 1) #X Axis seconds
        ticks = list(self.eoi_data_df.columns) 
        num_columns = len(self.eoi_data_df.columns) 
        xticks = np.linspace(0,num_columns-1,num_secs, dtype = int)
        xticklabels = [int(ticks[i]) for i in xticks]
        fig,axes = plt.subplots(figsize = (24,12))
        sns.heatmap(self.eoi_data_df, cmap = 'viridis') #Plot data via Seaborn
        plt.xticks(ticks= xticks,labels = xticklabels)
        plt.xlabel('Time (Seconds)', fontsize = 24)
        plt.ylabel('Trial (# w/n Session)', fontsize = 24)
        plt.title(eoi_type.upper(), fontsize = 24)
        plt.tick_params(labelsize = 16)
        plt.tight_layout()
        plt.savefig(f'C:\\PATH TO FIGURE SAVE LOCATION\\Figures\\{get_epocs.data.info.blockname} {eoi_type.upper()} Heatmap.png', dpi = 600) #Leave in to save heatmap to given location
    

    #Must-call for the following two methods (they are dependent on the creation of average_df)
    def avg_df_maker(self):
        self.average_df = pd.DataFrame(self.eoi_data_df.mean())
        self.average_df.reset_index(inplace=True)
        self.average_df.rename(columns = {'index': 'Time (Seconds)', 0: 'Avg. Z-Score'}, inplace = True)


    #Lineplot of average signal across all trails centered around the epoc event
    def avg_lineplt(self):
        average_df = pd.DataFrame(self.eoi_data_df.mean())
        average_df.reset_index(inplace=True)
        average_df.rename(columns = {'index': 'Time (Seconds)', 0: 'Avg. Z-Score'}, inplace = True)

        plt.figure(figsize = (24,18))
        plt.plot(self.average_df['Time (Seconds)'], self.average_df['Avg. Z-Score'],)
        plt.plot(self.pre, 0.5, marker = 'o', markersize= 20.0)
        plt.xlabel('Time (Seconds)', fontsize = 24)
        plt.ylabel('Avg. Normalized F/f (Z-Score)' , fontsize = 24)
        plt.title(f'AVERAGE {eoi_type.upper()}', fontsize= 24)
        plt.tick_params(labelsize = 20)
        plt.tight_layout()
        plt.savefig(f'C:\\PATH TO FIGURE SAVE LOCATION\\Figures\\Average {eoi_type} Traces for {self.blockname}.png', dpi = 600)
    

    #All traces centered around the epoc event
    def all_epoc_traces(self):
        fig,axes = plt.subplots(figsize = (24,18))
        for i in range(len(self.eoi_data_df)):
            plt.plot(self.eoi_data_df.iloc[i])
        plt.plot(self.pre, 0.5, marker = 'o', markersize= 30.0, markerfacecolor = 'r')
        plt.xlabel('Time (Seconds)', fontsize = 24)
        plt.ylabel('Avg. Normalized F/f (Z-Score)', fontsize = 24)
        plt.title(f'ALL {eoi_type.upper()} TRACES', fontsize = 24)
        plt.tick_params(labelsize = 20)
        plt.tight_layout()
        plt.savefig(f'C:\\PATH TO FIGURE SAVE LOCATION\\Figures\\All {eoi_type} Traces for {self.blockname}.png', dpi = 600)

        

get_epocs = All_Epocs(path_to_file, epoc_name, eoi_port)
get_epocs.make_epoc_df()
get_epocs.isolate_epoc_type()



analyze_eoi= EOI_Tools(path_to_file, exp_ch_name, control_ch_name, time_pre_epoc, time_post_epoc, get_epocs.eoi_df, get_epocs.data.info.blockname)
analyze_eoi.data_by_epoc()
analyze_eoi.heatmapper()
analyze_eoi.avg_df_maker()
analyze_eoi.avg_lineplt()
analyze_eoi.all_epoc_traces()
