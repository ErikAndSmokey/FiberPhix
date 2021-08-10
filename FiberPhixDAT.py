#The Art Vandelay Industries imports/exports

import tdt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import signal
from scipy import stats
import time


entry_df= pd.ExcelFile('PATH TO USER ENTRY SPREADSHEET').parse()
path_to_data = str(entry_df['Path to tank file'][0])
start_of_stream_section = int(entry_df['Time to cut off beginning (seconds)'][0])
end_of_stream_section = int(entry_df['Time to cut off end (seconds)'][0])
#lowess_fraction = entry_df['LOWESS Fraction***'][0]
lowess_fraction= 0.001
exp_ch_name = str(entry_df['Signal Channel ID'][0])
control_ch_name = str(entry_df['Control Channel ID'][0])



#Create a class for gathering, detrending, and fitting any given stream data
class StreamData():

	def __init__(self, data_file_path, start_cut, end_cut, lowess_data_frac, stream_name):
		self.path = data_file_path
		self.start = start_cut
		self.end = end_cut
		self.lowess_frac = lowess_data_frac
		self.stream_name = stream_name
		self.raw_data = tdt.read_block(self.path,t1 = self.start, t2 = self.end)


	def clear_stream(self):
		stream_data = self.raw_data.streams[self.stream_name]['data']
		self.stream_time = (np.linspace(1,len(stream_data), len(stream_data))/self.raw_data.streams[self.stream_name].fs) + self.start
		stream_detrend = signal.detrend(stream_data)
		stream_coefs = np.polyfit(stream_detrend, stream_data,1)
		self.stream_data = [stream_coefs[0]*i+stream_coefs[1] for i in stream_detrend]


#The main class for assessing the signal vs control streams... will not cause total protonic reversal
#Makes a Pandas DataFrame and performs calculations necessary for a delta f/f value
class CrossTheStreams():
    
    def __init__(self,exp_stream_data,control_stream_data,time_values):
        self.exp_data = exp_stream_data
        self.control_data = control_stream_data
        self.time = time_values
    
    def stream_comp_df(self):
        comp_df_builder = {'Time': self.time,
                          'Exp Signal': self.exp_data,
                          'Control Signal': self.control_data}
        self.df = pd.DataFrame(comp_df_builder)
        
    def delta_f_over_f(self):
        scale_coefs = np.polyfit(self.df['Control Signal'], self.df['Exp Signal'],1)
        self.df['Scaled Control'] = [scale_coefs[0]*i+scale_coefs[1] for i in self.df['Control Signal']]
        self.df['Exp - Control'] = self.df['Exp Signal'] - self.df['Scaled Control']
        self.df['F/F'] = self.df['Exp - Control']/self.df['Scaled Control']
        self.df['Normalized F/F (Z-Score)'] = stats.zscore(self.df['F/F'])


def main():
	#Create your signal stream object
	exp_stream = StreamData(path_to_data,start_of_stream_section,end_of_stream_section,lowess_fraction,exp_ch_name)
	exp_stream.clear_stream()

	#Create your control stream object
	control_stream = StreamData(path_to_data,start_of_stream_section,end_of_stream_section,lowess_fraction,control_ch_name)
	control_stream.clear_stream()

	exp_v_control = CrossTheStreams(exp_stream.stream_data,control_stream.stream_data,exp_stream.stream_time)
	exp_v_control.stream_comp_df()
	exp_v_control.delta_f_over_f()


	fig,axes = plt.subplots(2,2, figsize = (24,12))
	axes[0,0].plot(exp_v_control.df['Time'],exp_v_control.df['Exp Signal'],'g')
	axes[0,0].plot(exp_v_control.df['Time'],exp_v_control.df['Control Signal'],'m')
	axes[0,0].set_ylabel('Raw Signal (Detrended)')
	axes[0,0].set_xlabel('Time (Seconds)')
	axes[0,1].plot(exp_v_control.df['Time'],exp_v_control.df['Exp Signal'],'g')
	axes[0,1].plot(exp_v_control.df['Time'],exp_v_control.df['Scaled Control'],'m')
	axes[0,1].set_ylabel('Fitted Signals (Control Fit to Exp)')
	axes[0,1].set_xlabel('Time (Seconds)')
	axes[1,0].plot(exp_v_control.df['Time'],exp_v_control.df['F/F'],'g')
	axes[1,0].set_ylabel('F/F (Exp from Control)')
	axes[1,0].set_xlabel('Time (Seconds)')
	axes[1,1].plot(exp_v_control.df['Time'],exp_v_control.df['Normalized F/F (Z-Score)'],'g')
	axes[1,1].set_ylabel('Normalized F/F (Z-Score)')
	axes[1,1].set_xlabel('Time (Seconds)')
	fig.tight_layout()
	plt.savefig('PATH TO FOLDER FOR FIGURE STORAGE',dpi = 600)


if __name__ == '__main__':
	main()