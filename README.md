# FiberPhix
Python fiber photometry data analysis tool. READS TDT TANK FILES.


#FiberPhixDAT_MainInfo: pulls all the main stream and epoc info (i.e., the names of the streams/epoc) for entry into the 'user values' spreadsheet (GUI to come...).

#FiberPhixDAT: In the 'User Values' spreadsheet, a user defines the signal channel and control channel (typically isosbestic channel). FiberPhixDAT then measures
#the change in signal vs. isosbestic. The 'User Values' spreadsheet also allows for the adjustment of window size so that the beginning and ends of sessions can be cut off.
#Data is plotted via matplotlib at 600 dpi. User may define different graphs, default setting is 4 graphs.

FiberPhixDAT_Epocs: Using the 'User Values' spreadsheet, one can identify the epocs of interest, window centering size (time pre/post epoc). Ultimately collects data,
#analyzes (using similar methods to FiberPhixDAT) and graphs using a seaborn heatmap for easy visualization.

#Things the user must change:
1) file paths for 'User Values' spreadsheet
2) output file paths for graphs
3) out file path for FiberPhixDAT_MainInfo (where you'll store the available session stream/epoc ID's)
4) the 'User Values' spreadsheet for the relevant measures



#Suggested file structure:
(Root)
-(File) FiberPhixDAT
-(File) FiberPhixDAT_MainInfo
-(File) FiberPhixDAT_Epocs
-(Folder) Figures <--- Holds graphs generated from above
-(Folder) Data <--- Where you will add the tank file of interest
-(Folder) IDs <--- Where the excel file from 'FiberPhixDAT_MainInfo' will be created
