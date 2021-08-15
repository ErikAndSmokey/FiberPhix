**FiberPhix**


Python fiber photometry data collection tool. READS TDT TANK FILES. 



**-FiberPhixDAT_MainInfo**: pulls all the main stream and epoc info (i.e., the names of the streams/epoc) for entry into the 'user values' spreadsheet. ***NEW*** Exported spreadsheet includes all epoc information for user comprehension of acquired data.


***NEW*** FiberPhixDAT_BatchMainInfo: Same functionality as _MainInfo_, but allows for iteration over multiple tank files and produces an information spreadsheet for each. 


**-FiberPhixDAT**: In the 'User Values' spreadsheet, a user defines the signal channel and control channel (typically isosbestic channel). FiberPhixDAT then measures
the change in signal vs. isosbestic. The 'User Values' spreadsheet also allows for the adjustment of window size so that the beginning and ends of sessions can be cut off.
Data is plotted via matplotlib at 600 dpi.


**FiberPhixDAT_Epocs**: Using the 'User Values' spreadsheet, one can identify the epocs of interest, signal vs control channels, and window centering size (time pre/post epoc). Ultimately collects data from epocs of interest (EOI), analyzes (using methods dev'd in FiberPhixDAT) and graphs using a seaborn heatmap for easy visualization of a normalized
delta f/f. ***NEW*** now plots an 'average' trace for all events that match the epoc of interest (EOI). Additionally plots all traces centered around the user-defined epoc window (time pre/post).


**User Values Spresheet**: Currently used as the place to adjust values for FiberPhixDAT, such as identifying the signal stream, the control stream, time to cut off, etc. This spreadsheet
is used by all scripts.


**Things the user must change**:
1) File paths for 'User Values' spreadsheet
2) Output file paths for graphs
3) Output file path for FiberPhixDAT_MainInfo (where you'll store the available session stream/epoc ID's) <--- useful for filling in your 'User Values' spreadsheet
4) 'User Values' spreadsheet (row 2) for the relevant measures. Modifying header names will require editing scripts to match column headers.



**Suggested file structure**:
(Root)

-(File) FiberPhixDAT.py

-(File) FiberPhixDAT_MainInfo.py

***NEW***-(File) FiberPhixDAT_BatchMainInfo.py

-(File) FiberPhixDAT_Epocs.py

-(File) User Values.xlsx

-(Folder) Figures <--- Holds graphs generated from above

-(Folder) Data <--- Where you will add the tank file(s) of interest

-(Folder) IDs <--- Where the excel file from 'FiberPhixDAT_MainInfo' will be created
