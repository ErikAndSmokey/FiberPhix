import tdt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import signal
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d


root_folder = os.getcwd()
PATH_TO_USER_VALUES = root_folder + '\\User Values.xlsx'
data_folder = root_folder + '\\Data'
ids_folder = root_folder + '\\IDs'
figures_folder = root_folder + '\\Figures'


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



def getinfo():

    entry_df= pd.ExcelFile(PATH_TO_USER_VALUES).parse()
    path_to_data = data_folder + '\\' + str(entry_df['Tank File Name'][0])
    exp_ch_name = str(entry_df['Signal Channel ID'][0])
    control_ch_name = str(entry_df['Control Channel ID'][0])
    infosaveloc = ids_folder


    #Main --> gathers data from both  streams/epocs, converts to pandas dataframe and exports to excel        
    information = Main_Info(path_to_file= path_to_data)
    information.get_main_info()

    epocs = Get_Epocs(epoc_info= information.epocs)
    epocs.all_epoc_info()

    stream_id_df = pd.DataFrame([[list(information.streams.keys()),information.session_length]], 
                                columns = [['Stream IDs Available', 'Session Length (Seconds)']])
    epoc_df = pd.DataFrame(epocs.df_builder)

    with pd.ExcelWriter(infosaveloc + f'\\Information from {information.blockname}.xlsx') as writer:
        stream_id_df.to_excel(writer, sheet_name = 'Stream IDs')
        for i in epocs.epoc_names:
            df = pd.DataFrame(list(zip(epocs.epoc_info[i]['data'], epocs.epoc_info[i]['onset'], epocs.epoc_info[i]['offset'])),
                              columns = ['Port', 'Onsets', 'Offsets'])
            df.to_excel(writer, sheet_name = f'EPOC {i}')

def batchinfo():

    entry_df= pd.ExcelFile(PATH_TO_USER_VALUES).parse()
    path_to_data = data_folder +'\\' + str(entry_df['Tank File Name'][0])
    exp_ch_name = str(entry_df['Signal Channel ID'][0])
    control_ch_name = str(entry_df['Control Channel ID'][0])
    infosaveloc = ids_folder
    dir_for_batch_info = data_folder

    for i in os.listdir(dir_for_batch_info):
        information = Main_Info(path_to_file= (str(dir_for_batch_info + '/' + i)))
        information.get_main_info()

        epocs = Get_Epocs(epoc_info= information.epocs)
        epocs.all_epoc_info()

        stream_id_df = pd.DataFrame([[list(information.streams.keys()),information.session_length]],
                                     columns = [['Stream IDs Available', 'Session Length (Seconds)']])
        epoc_df = pd.DataFrame(epocs.df_builder)

        with pd.ExcelWriter(infosaveloc+ f'\\Information from {information.blockname}' + '.xlsx') as writer:
            stream_id_df.to_excel(writer, sheet_name = 'Stream IDs')
            for i in epocs.epoc_names:
                df = pd.DataFrame(list(zip(epocs.epoc_info[i]['data'], epocs.epoc_info[i]['onset'], epocs.epoc_info[i]['offset'])),
                                  columns = ['Port', 'Onsets', 'Offsets'])
                df.to_excel(writer, sheet_name = f'EPOC {i}')





#Begin script section for FiberPhix.streams()

class StreamData():

    def __init__(self, data_file_path, start_cut, end_cut, stream_name):
        self.path = data_file_path
        self.start = start_cut
        self.end = end_cut
        self.stream_name = stream_name
        self.raw_data = tdt.read_block(self.path,t1 = self.start, t2 = self.end)
        self.blockname = self.raw_data.info.blockname

        #Gather the stream data, create a timeline, detrend signal, and refit signal to original plot
    def clear_stream(self):
        stream_data = self.raw_data.streams[self.stream_name]['data']
        self.stream_time = (np.linspace(1,len(stream_data), len(stream_data))/self.raw_data.streams[self.stream_name].fs) + self.start
        stream_detrend = signal.detrend(stream_data)
        stream_coefs = np.polyfit(stream_detrend, stream_data,1)
        self.stream_data = [stream_coefs[0]*i+stream_coefs[1] for i in stream_detrend]

class CrossTheStreams():
    
    def __init__(self,exp_stream_data,control_stream_data,time_values,figsaveloc,blockname,window_size):
        self.exp_data = exp_stream_data
        self.control_data = control_stream_data
        self.time = time_values
        self.figsaveloc = figsaveloc
        self.blockname = blockname
        self.window_size = window_size

    #Take your signal vs control stream and combine into a signal DataFrame
    def stream_comp_df(self):
        comp_df_builder = {'Time': self.time,
                          'Exp Signal': self.exp_data,
                          'Control Signal': self.control_data}
        self.df = pd.DataFrame(comp_df_builder)
      #perform signal vs control analysis ---> z-score delta f/f  
    def delta_f_over_f(self):
        self.df['Sliding Average Control'] = uniform_filter1d(self.df['Control Signal'], size = int(self.window_size*1000), mode = 'nearest')
        self.df['Sliding Average Exp'] = uniform_filter1d(self.df['Exp Signal'], size = int(self.window_size*1000), mode = 'nearest')
        sa_coefs = np.polyfit(self.df['Sliding Average Control'], self.df['Sliding Average Exp'],1)
        self.df['Scaled SA Control'] = [sa_coefs[0]*i+sa_coefs[1] for i in self.df['Sliding Average Control']]
        self.df['SA Exp - Scaled SA Control'] = self.df['Sliding Average Exp'] - self.df['Sliding Average Control']
        self.df['df/f'] = self.df['SA Exp - Scaled SA Control']/self.df['Sliding Average Control']
        self.df['Normalized df/f (Z-Score)'] = stats.zscore(self.df['df/f'])

    def graph_streams(self):
        fig,axes = plt.subplots(2,2, figsize = (24,12))
        axes[0,0].plot(self.df['Time'],self.df['Sliding Average Exp'],'g')
        axes[0,0].plot(self.df['Time'],self.df['Sliding Average Control'],'m')
        axes[0,0].set_ylabel('Raw Signal (Detrended)')
        axes[0,0].set_xlabel('Time (Seconds)')
        axes[0,1].plot(self.df['Time'],self.df['Sliding Average Exp'],'g')
        axes[0,1].plot(self.df['Time'],self.df['Scaled SA Control'],'m')
        axes[0,1].set_ylabel('Fitted Signals (Control Fit to Exp)')
        axes[0,1].set_xlabel('Time (Seconds)')
        axes[1,0].plot(self.df['Time'],self.df['df/f'],'g')
        axes[1,0].set_ylabel('F/F (Exp from Control)')
        axes[1,0].set_xlabel('Time (Seconds)')
        axes[1,1].plot(self.df['Time'],self.df['Normalized df/f (Z-Score)'],'g')
        axes[1,1].set_ylabel('Normalized F/F (Z-Score)')
        axes[1,1].set_xlabel('Time (Seconds)')
        fig.tight_layout()
        plt.savefig(self.figsaveloc + f'\\Stream graphs for {self.blockname} .png',dpi = 600)





def streams():       
    entry_df= pd.ExcelFile(PATH_TO_USER_VALUES).parse()
    path_to_data = data_folder + '\\' + str(entry_df['Tank File Name'][0])
    exp_ch_name = str(entry_df['Signal Channel ID'][0])
    control_ch_name = str(entry_df['Control Channel ID'][0])
    start_of_stream_section = int(entry_df['Start stream collection at (seconds)'][0])
    end_of_stream_section = int(entry_df['End stream collection at (seconds)'][0])
    window_size = float(entry_df['Sliding average window size (Seconds)'][0])
    figsaveloc = figures_folder


    #Create your signal stream object
    exp_stream = StreamData(data_file_path = path_to_data,
                                start_cut = start_of_stream_section,
                                end_cut = end_of_stream_section,
                                stream_name = exp_ch_name)
    exp_stream.clear_stream()

    #Create your control stream object
    control_stream = StreamData(data_file_path= path_to_data,
                                    start_cut = start_of_stream_section,
                                    end_cut = end_of_stream_section,
                                    stream_name = control_ch_name)
    control_stream.clear_stream()

    exp_v_control = CrossTheStreams(exp_stream_data= exp_stream.stream_data,
                                        control_stream_data= control_stream.stream_data,
                                        time_values=exp_stream.stream_time,
                                        figsaveloc=figsaveloc, 
                                        blockname= exp_stream.blockname,
                                       window_size = window_size)
    exp_v_control.stream_comp_df()
    exp_v_control.delta_f_over_f()
    exp_v_control.graph_streams()

        


#Begin script section for FiberPhix.epocs()

class All_Epocs():

    def __init__(self, path_to_file,epoc_name,eoi_port,eoi_pattern):
        self.path = path_to_file
        self.name = epoc_name
        self.data = tdt.read_block(self.path)
        self.eoi = eoi_port
        self.eoi_pattern = eoi_pattern
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
        elif str(self.eoi).upper() == 'PATTERN':
             self.eoi_df = self.all_epocs_df[::self.eoi_pattern]
        else:
            self.eoi_df = self.all_epocs_df[self.all_epocs_df['Epoc Data'] == self.eoi]



#After isolating your epoc type of interest    
class EOI_Tools():
    
    def __init__(self,path_to_file, exp_ch, control_ch, pre_epoc_time, post_epoc_time, eoi_df, blockname, eoi_type, figsaveloc):
        self.exp = exp_ch
        self.control = control_ch
        self.pre = pre_epoc_time
        self.post = post_epoc_time
        self.path = path_to_file
        self.eoi_df = eoi_df
        self.onsets = list(self.eoi_df['Onsets'])
        self.blockname = blockname
        self.eoi_type = eoi_type
        self.figsaveloc = figsaveloc
        self.window_size = (self.pre+self.post)*0.001
        
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
            sa_coefs = np.polyfit(temp_epoc_df['Control Signal'], temp_epoc_df['Exp Signal'],1)
            temp_epoc_df['Scaled Control'] = [sa_coefs[0]*i+sa_coefs[1] for i in temp_epoc_df['Control Signal']]
            temp_epoc_df['Exp Signal - Scaled Control'] = temp_epoc_df['Exp Signal'] - temp_epoc_df['Scaled Control']
            temp_epoc_df['df/f'] = temp_epoc_df['Exp Signal - Scaled Control']/temp_epoc_df['Scaled Control']
            temp_epoc_df['Normalized df/f (Z-Score)'] = stats.zscore(temp_epoc_df['df/f'])
            
            
            
            
            
            self.data.append(list(temp_epoc_df['Normalized df/f (Z-Score)'][:self.min_time]))
        
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
        num_secs_pre  = list([i*-1 for i in range(0,self.pre+1)])
        num_secs_post = list(range(1,self.post+1))
        xticklabels = num_secs_pre + num_secs_post
        xticklabels.sort()
        fig,axes = plt.subplots(figsize = (24,12))
        sns.heatmap(self.eoi_data_df, cmap = 'viridis') #Plot data via Seaborn
        plt.xticks(ticks= xticks,labels = xticklabels)
        plt.xlabel('Time (Seconds)', fontsize = 24)
        plt.ylabel('Trial (# w/n Session)', fontsize = 24)
        plt.title(self.eoi_type.upper(), fontsize = 24)
        plt.tick_params(labelsize = 16)
        plt.tight_layout()
        plt.savefig(self.figsaveloc + f'\\{self.blockname} {self.eoi_type.upper()} Heatmap.png', dpi = 600) #Leave in to save heatmap to given location
    

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
        
        num_secs = (self.pre + self.post + 1)
        ticks = list(self.average_df['Time (Seconds)']) 
        num_columns = len(self.average_df['Time (Seconds)']) 
        xticks = np.linspace(0,num_columns-1,num_secs, dtype = int)
        xtickspots = [int(ticks[i]) for i in xticks]
        num_secs_pre  = list([i*-1 for i in range(0,self.pre+1)])
        num_secs_post = list(range(1,self.post+1))
        xticklabels = num_secs_pre + num_secs_post
        xticklabels.sort()
        
        
        plt.figure(figsize = (24,18))
        plt.plot(self.average_df['Time (Seconds)'], self.average_df['Avg. Z-Score'],)
        plt.plot(self.pre, 0.5, marker = 'o', markersize= 20.0)
        plt.xlabel('Time (Seconds)', fontsize = 24)
        plt.ylabel('Avg. Normalized F/f (Z-Score)' , fontsize = 24)
        plt.title(f'AVERAGE {self.eoi_type.upper()}', fontsize= 24)
        plt.xticks(ticks = xtickspots, labels = xticklabels)
        plt.tick_params(labelsize = 20)
        plt.tight_layout()
        plt.savefig(self.figsaveloc+ f'\\Average {self.eoi_type} Traces for {self.blockname}.png', dpi = 600)
    

    #All traces centered around the epoc event
    def all_epoc_traces(self):
        fig,axes = plt.subplots(figsize = (24,18))
        for i in range(len(self.eoi_data_df)):
            plt.plot(self.eoi_data_df.iloc[i])
        plt.plot(self.pre, 0.5, marker = 'o', markersize= 30.0, markerfacecolor = 'r')
        plt.xlabel('Time (Seconds)', fontsize = 24)
        plt.ylabel('Avg. Normalized F/f (Z-Score)', fontsize = 24)
        plt.title(f'ALL {self.eoi_type.upper()} TRACES', fontsize = 24)
        plt.tick_params(labelsize = 20)
        plt.tight_layout()
        plt.savefig(self.figsaveloc + f'\\All {self.eoi_type} Traces for {self.blockname}.png', dpi = 600)





def epocs():
    entry_df= pd.ExcelFile(PATH_TO_USER_VALUES).parse()
    path_to_data = data_folder + '\\' + str(entry_df['Tank File Name'][0])
    exp_ch_name = str(entry_df['Signal Channel ID'][0])
    control_ch_name = str(entry_df['Control Channel ID'][0])
    time_pre_epoc = int(entry_df['Time to grab before each epoc (seconds)'][0])
    time_post_epoc = int(entry_df['Time to grab after each epoc (seconds)'][0])
    epoc_name = str(entry_df['Epoc ID'][0])
    eoi_port = entry_df['Epoc Port to grab?'][0] #The epoc type of interest... for centering around that type of epoc later
    eoi_type = entry_df['Epoc Type?'][0]
    figsaveloc = figures_folder
    eoi_pattern = entry_df['Epoc Port to grab?'][2]
    #window_size = float(entry_df['Sliding average window size (Seconds)'][0])


    get_epocs = All_Epocs(path_to_file = path_to_data, 
                            epoc_name = epoc_name, 
                            eoi_port = eoi_port,
                         eoi_pattern = eoi_pattern)


    get_epocs.make_epoc_df()
    get_epocs.isolate_epoc_type()
    analyze_eoi= EOI_Tools(path_to_file = path_to_data, 
                            exp_ch = exp_ch_name, 
                            control_ch = control_ch_name, 
                            pre_epoc_time = time_pre_epoc, 
                            post_epoc_time = time_post_epoc, 
                            eoi_df = get_epocs.eoi_df, #Note that this is generated by the previous function
                            blockname = get_epocs.data.info.blockname, #Note that this is generated by the previous function
                            eoi_type = eoi_type, 
                            figsaveloc = figsaveloc)

    analyze_eoi.data_by_epoc()
    analyze_eoi.heatmapper()
    analyze_eoi.avg_df_maker()
    analyze_eoi.avg_lineplt()