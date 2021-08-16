**FiberPhix**


Python fiber photometry data collection tool. READS TDT TANK FILES. 



***NEW*** **FiberPhix.getinfo()**: Pulls all the stream and epoc information (i.e., the names of the streams/epoc) and organizes the information into a highly-readable excel spreadsheet for the user. The information collected can assist in the completion of the 'User Values' spreadsheet if parameters are unknown (ex., identifying what epoc ports are important and available on each epoc channel).


***NEW*** **FiberPhix.batchinfo()**: Same functionality as .getinfo(), but allows for iteration over multiple tank files, producing an information spreadsheet for each tank. 


***NEW*** **FiberPhix.streams()**: In the 'User Values' spreadsheet, a user defines the signal channel and control channel (typically isosbestic channel). FiberPhix then measures the change in signal vs. isosbestic within the window size that the user defines in the spreadsheet. FiberPhix utilizes scipy.signal.detrend() to account for any photobleaching across the session and numpy.ployfit() for scaling the control signal. Data is plotted via matplotlib and exported as a PNG (at 600 dpi; adjustable w/n script).


***NEW*** **FiberPhix.epocs()**: Using the 'User Values' spreadsheet, one can identify the epocs of interest, signal & control channels, and window centering size (time pre/post epoc). Ultimately collects data from epocs of interest (EOI- i.e., the specific epoc 'data' value of interest), analyzes each individual epoc (via FiberPhix.streams()) and graphs using a seaborn heatmap for easy visualization of a normalized delta f/f (sns.heatmap()). Plots an 'average' trace for all events that match the epoc of interest (EOI). Additionally plots all traces centered around the user-defined epoc window (time pre/post).


***UPDATED*** **User Values Spresheet**: Currently used as the place to adjust values for FiberPhix, such as identifying the signal stream, the control stream, time to cut off, etc. Editing values in this spreadsheet creates an updated list within 'FiberPhix' when imported. DO NOT MOVE COLUMNS.


***UPDATED*** **Things the user must change**:
1) File paths for 'User Values' spreadsheet (edit in FiberPhix py script directly)
2) 'User Values' spreadsheet (row 2) as indicated by headers. Modifying header names will require editing scripts to match column headers.



***UPDATED!*** **Suggested file structure**:
(Root)
-(File) FiberPhix.py

-(File) User Values.xlsx

-(Folder) Figures <--- Holds graphs generated from 'FiberPhix.streams()' and 'FiberPhix.epocs()'

-(Folder) Data <--- Where you will add the tank file(s) of interest

-(Folder) IDs <--- Where the excel file(s) from 'FiberPhix.getinfo()' or 'FiberPhix.batchinfo()' will be saved
