from tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
	root = Tk()
	root.title("Home page")

	width= root.winfo_screenwidth() 
	height= root.winfo_screenheight()
	root.geometry("%dx%d" % (width, height))
	root.state('zoomed')
	
	Label(root,text="We are going to use notebook widget\nSave tables in a csv file\nLoad it in a specific window \
		\nYou can probably use panned window if not notebook").pack()

	root.mainloop()