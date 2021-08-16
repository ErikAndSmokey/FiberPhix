**FiberPhix**


Python fiber photometry data collection tool. READS TDT TANK FILES. 



***NEW*** **FiberPhix.getinfo()**: pulls all the main stream and epoc info (i.e., the names of the streams/epoc) for entry into the 'user values' spreadsheet. ***NEW*** Exported spreadsheet includes all epoc information for user comprehension of acquired data.


***NEW*** **FiberPhix.batchinfo()**: Same functionality as _MainInfo_, but allows for iteration over multiple tank files, producing an information spreadsheet for each tank. 


***NEW*** **FiberPhix.streams()**: In the 'User Values' spreadsheet, a user defines the signal channel and control channel (typically isosbestic channel). FiberPhixDAT then measures
the change in signal vs. isosbestic. The 'User Values' spreadsheet also allows for the adjustment of window size so that the beginning and ends of sessions can be cut off.
Data is plotted via matplotlib at 600 dpi.


***NEW*** **FiberPhix.epocs()**: Using the 'User Values' spreadsheet, one can identify the epocs of interest, signal vs control channels, and window centering size (time pre/post epoc). Ultimately collects data from epocs of interest (EOI), analyzes (using methods dev'd in FiberPhixDAT) and graphs using a seaborn heatmap for easy visualization of a normalized delta f/f. Plots an 'average' trace for all events that match the epoc of interest (EOI). Additionally plots all traces centered around the user-defined epoc window (time pre/post).


***UPDATED*** **User Values Spresheet**: Currently used as the place to adjust values for FiberPhixDAT, such as identifying the signal stream, the control stream, time to cut off, etc. Editing values in this spreadsheet creates an updated list within 'FiberPhix' when imported.


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
