#!/usr/bin/env python
#=============================================================================
#   Copyright (c) 2019 by Mark A. Abramson
#
#   getConfoundingInterval.py is free software; you can redistribute it and/or
#   modify it under the terms of the GNU General Public License as published 
#   by the Free Software Foundation; either version 3 of the License, or 
#   (at your option) any later version.
#
#   getConfoundingInterval.py is distributed in the hope that it will be
#   useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#   Public License for more details.  A copy of the GNU General Public License
#   is available at
#   http://www.gnu.org/licenses/gpl-3.0.en.html.
#=============================================================================

import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ConfoundingInterval import *

#=============================================================================
# MainApplication:  Class for the main application of the GUI
#-----------------------------------------------------------------------------
#  VARIABLES:
#   ftb          = default bold font for Entry fields
#   prob_frame   = frame that holds the Problem menu and its label
#   prob_menu    = options menu for test problem selection
#   prob_text    = label for Problem options menu
#   prob_choice  = list of possible choices in Problem options menu
#   prob         = variable holding current choice in Problem options menu
#   alg_frame    = frame that holds the Algorithm Options Menu and its label
#   alg_menu     = options menu for selecting the optimization algorithm
#   alg_text     = label for Algorithms options menu
#   alg_choice   = list of possible choices in Algorithms options menu
#   alg          = variable that tracks the currently selected optimizer
#   vcmd         = function handle for validating data in an Entry field
#   st_in        = vector of horizontal alignments of input  column headers
#   st_out       = vector of horizontal alignments of output column headers
#   n1/n2        = lists of '-' or '' used for input data validation
#   d_headers    = column headers for measured data parameters
#   b_headers    = column headers for lower and upper bounds
#   d_labels     = name and range for each data parameter
#   b_labels     = name and range for each pair of variable bounds
#   d_tag        = array of strings that identify a data field with bad data
#   b_tag        = array of strings that identify a bound field with bad data
#   input_frame  = frame that holds all data input fields and their labels
#   txtTitle     = plot title text
#   txtX         = x-axis label text
#   txtY         = y-axis label text
#   txtLegend    = legend text
#   c            = temporary storage of current tkinter class widget
#   var          = vector of Entry objects for data parameters
#   lb           = vector of Entry objects for variable lower bounds
#   ub           = vector of Entry objects for variable upper bounds
#   d_var        = vector of variables that track content of var
#   lb_var       = vector of variables that tracks content of lb
#   ub_var       = vector of variables that tracks content of ub
#   pb_frame     = frame that holds the push button objects
#   pb           = array of push button-specific data
#   err_msg      = variable that holds error message
#   err_text     = label that holds the error message
#   out_frame    = frame that holds all the output fields
#   out_title    = label that holds the title of the output table
#   o_headers    = list of text output table column headers
#   o_ci         = list of taxt labels for confounding interval bounds
#   out_header   = list of labels for confounding interval solution
#   out_interval = list of data for the confounding interval solution
#   min          = list of data for each parameter that yields CI lower bound
#   max          = list of data for each parameter that yields CI upper bound
#   canvas       = GUI canvas on which plots are created
#-----------------------------------------------------------------------------
#  METHODS:
#   load_problem    = loads selected problem parameter into GUI Entry fields
#   set_error_color = process a GUI Entry input error
#   load_parameters = loads GUI Entry field values into variables
#   clear_plot      = clears the current plot
#   validate_entry  = restricts EntryTable entry fields to be floating point
#   problem_cb      = Problem options menu callback function
#   reset_cb        = Reset pushbutton callback function
#   solve_cb        = Solve pushbotton callback function
#   plot_cb         = Plot  pushbutton callback function
#=============================================================================
class MainApplication(tk.Frame):

    # Constructor
    def __init__(self,parent,*args,**kwargs):
        tk.Frame.__init__(self,parent,*args,**kwargs)

        self.parent = parent
        ftb = 'Helvetica 11 bold'

        # Popup menu for choosing a problem from a library
        prob_frame = tk.Frame(parent)
        self.prob_choice = [*TestProblemLibrary.keys()]
        self.prob = tk.StringVar()
        prob_text = tk.Label(prob_frame, text='Preset Problem:',font=ftb)
        prob_menu = tk.OptionMenu(prob_frame,self.prob,*self.prob_choice)
        prob_menu.configure(bg='purple',fg='white',font=ftb,
                            width=len(max(self.prob_choice))+1)
        prob_text.pack(side=tk.LEFT)
        prob_menu.pack(side=tk.LEFT)
        self.prob.trace('w', self.problem_cb)

        #===================== BEGIN Input Table subframe ====================

        # Local variables
        vcmd   = self.register(self.validate_entry)
        st_in  = [tk.SW, tk.S, tk.S, tk.S, tk.S]
        st_out = [tk.SW, tk.S, tk.E, tk.E]
        n1     = ['-','','']
        n2     = ['','','-']

        # Text Labels and data used in the GUI
        d_labels = [['Correlation Coefficient',u'\u03C1(x,y)','(-1.0, 1.0)'],
                    ['Standard Deviation in y',u'\u03C3(y)', u'\u2265 0'],
                    ['Standard Deviation in x',u'\u03C3(x)', u'\u2265 0'] ]
        b_labels = [['Coefficient of Determination of w on x',
                     u'R\u00B2(w,x)','[ 0.0, 1.0)'],
                    ['Coefficient of Determination of w on y',
                     u'R\u00B2(w,y)','[ 0.0, 1.0)'],
                    ['Correlation Between Fitted Vectors',
                     u'\u03C1(x\u0302,y\u0302)','[-1.0, 1.0]']]
        self.d_tag = ['rho', 'sigma(y)', 'sigma(x)']
        self.b_tag = ['Parameter 1','Parameter 2','Parameter 3']

        # Plot label strings
        self.txt_title  = 'Confounding Interval'
        self.txt_y      = 'Adjusted Slope Coefficient' + r' $\beta_{x | w}$'
        self.txt_x      = b_labels[2][0] + r' $ \rho_{\hat{x}\hat{y}}$'
        self.txt_legend = ('Lower Bound','Upper Bound')

        # Column headers for measured rho, sx, sy data
        input_frame = tk.Frame(parent)
        d_headers = ['Measured Statistics','Symbol','Value','Range']
        for j in range(len(d_headers)):
            c = tk.Label(input_frame,text=d_headers[j],font=ftb)
            c.grid(row=1,column=j,pady=(5,0),sticky=st_in[j])

        # Labels and entry fields for measured rho, sx, sy data
        self.var = [None]*len(d_labels)
        self.d_var = [tk.StringVar() for k in range(len(d_labels))]
        for k in range(len(d_labels)):
            c = tk.Label(input_frame,text=d_labels[k][0]+': ')
            c.grid(row=k+2,column=0,sticky=tk.W)
            c = tk.Label(input_frame,text=d_labels[k][1])
            c.grid(row=k+2,column=1)
            self.var[k] = tk.Entry(input_frame,textvariable=self.d_var[k],
                             width=10,justify='center',validate='key',
                             validatecommand=(vcmd,n1[k],'%d','%P','%S'))
            self.var[k].grid(row=k+2,column=2,padx=5)
            c = tk.Label(input_frame,text=d_labels[k][2])
            c.grid(row=k+2,column=3)

        # Column headers for variable lower and upper bounds
        b_headers = ['Confounding Parameters','Symbol',
                     'Lower','Upper','Range']
        for j in range(len(b_headers)):
            c = tk.Label(input_frame,text=b_headers[j],font=ftb)
            c.grid(row=5,column=j,pady=(5,0),sticky=st_in[j])

        # Labels and entry fields for variable lower and upper bounds
        self.lb = [None]*len(b_labels)
        self.ub = [None]*len(b_labels)
        self.lb_var = [tk.StringVar() for k in range(len(b_labels))]
        self.ub_var = [tk.StringVar() for k in range(len(b_labels))]
        for k in range(len(b_labels)):
            c = tk.Label(input_frame,text=b_labels[k][0]+': ')
            c.grid(row=k+6,column=0,sticky=tk.W)
            tk.Label(input_frame,text=b_labels[k][1]).grid(row=k+6,column=1)
            self.lb[k]  = tk.Entry(input_frame,textvariable=self.lb_var[k],
                             width=10,justify='center',validate='key',
                             validatecommand=(vcmd,n2[k],'%d','%P','%S'))
            self.ub[k]  = tk.Entry(input_frame,textvariable=self.ub_var[k],
                             width=10,justify='center',validate='key',
                             validatecommand=(vcmd,n2[k],'%d','%P','%S'))
            self.lb[k].grid(row=k+6,column=2,padx=5)
            self.ub[k].grid(row=k+6,column=3,padx=5)
            tk.Label(input_frame,text=b_labels[k][2]).grid(row=k+6,column=4)
        #====================== END Input Table subframe =====================

        # Popup menu for choosing the optimization algorithm
        alg_frame = tk.Frame(parent)
        self.alg_choice = ['AOK','SLSQP']
        self.alg = tk.StringVar()
        alg_text = tk.Label(alg_frame,text="Optimizer:",font=ftb)
        alg_menu = tk.OptionMenu(alg_frame,self.alg,*self.alg_choice)
        alg_menu.configure(bg='purple',fg='white',font=ftb,
                          width=len(max(self.alg_choice))+2)
        self.alg.set(self.alg_choice[0])
        alg_text.pack(side=tk.LEFT)
        alg_menu.pack(side=tk.LEFT)

        # Subframe for pushbuttons
        pb = [['Reset','brown',self.reset_cb],
              ['Solve','green',self.solve_cb],
              ['Plot', 'blue', self.plot_cb ]]
        pb_frame = tk.Frame(parent)
        for j in range(len(pb)):
            c = tk.Button(pb_frame,text=pb[j][0],bg=pb[j][1],command=pb[j][2])
            c.grid(row=0,column=j,sticky=tk.EW)

        # Error message line
        self.err_msg  = tk.StringVar()
        self.err_msg.set('')
        self.err_text = tk.Label(parent,textvariable=self.err_msg,font=ftb,
                                 bg='red',fg='white')

        # Solution output (one line of text and a table)
        self.out_frame = tk.Frame(parent)
        out_title = tk.Label(self.out_frame,text=self.txt_title + ' Solution',
                             font=ftb)
        out_title.grid(row=0,column=0,columnspan=4,pady=5)
        o_headers  = ['Parameter','Symbol', 'Minimum', 'Maximum']
        o_ci       = ['Adjusted Slope Coefficient',u'\u03B2(x|w)','','']
        self.out_ci = [None]*len(o_headers)
        for j in range(len(o_headers)):
            c = tk.Label(self.out_frame,text=o_headers[j],font=ftb)
            c.grid(row=1,column=j,sticky=st_out[j])
            self.out_ci[j] = tk.Label(self.out_frame,text=o_ci[j],font=ftb,
                                      justify=tk.LEFT,bg='green',fg='white')
            self.out_ci[j].grid(row=2,column=j,sticky=st_out[j])

        self.min = [None]*len(b_labels)
        self.max = [None]*len(b_labels)
        for k in range(len(b_labels)):
            c = tk.Label(self.out_frame,text=b_labels[k][0] + ': ')
            c.grid(row=k+3,column=0,sticky=tk.W)
            c = tk.Label(self.out_frame,text=b_labels[k][1])
            c.grid(row=k+3,column=1,padx=5)
            self.min[k] = tk.Label(self.out_frame,text='')
            self.min[k].grid(row=k+3,column=2,sticky=tk.E)
            self.max[k] = tk.Label(self.out_frame,text='')
            self.max[k].grid(row=k+3,column=3,sticky=tk.E)

        # Pack the frames and load default values into GUI
        prob_frame.grid(row=0,column=0,pady=(5,0))
        input_frame.grid(row=1,column=0,pady=(5,0))
        alg_frame.grid(row=2,column=0,pady=(5,0))
        pb_frame.grid(row=3,column=0,pady=(5,0))
        self.err_text.grid(row=4,column=0,pady=(5,0))
        self.err_text.grid_remove()
        self.out_frame.grid(row=4,column=0,pady=(5,0))
        self.canvas = []
        self.load_problem('Default')

    #=========================================================================
    # load_problem:  Load a problem from the test problem library
    # ------------------------------------------------------------------------
    #  CALLED BY:  MainApplication, problem_cb, reset_cb
    #  CALLS:      clear_plot, set_error_color
    #  VARIABLES:
    #   choice = string matching the key of the selected library test problem
    #   p      = dictionary of parameter values from selected test problem
    #=========================================================================
    def load_problem(self,choice):
        self.prob.set(choice)
        p = TestProblemLibrary[choice]
        fields = ['rho','sy','sx']
        for k in range(len(self.d_var)):
            self.d_var[k].set(p[fields[k]])
        for k in range(len(self.lb_var)):
            self.lb_var[k].set(str(p['lb'][k]))
            self.ub_var[k].set(str(p['ub'][k]))
        self.err_msg.set('None')
        self.clear_plot()
        self.set_error_color()
        self.err_text.grid_remove()
        self.out_frame.grid_remove()

    #=========================================================================
    # set_error_color:  Set colors of a GUI Entry based on an error message
    #-------------------------------------------------------------------------
    #  CALLED BY:  load_problem, load_parameters
    #  VARIABLES:
    #   msg     = error message that appears on screen (could be none)
    #   fgcolor = cell foreground color
    #   bgcolor = cell background color
    #=========================================================================
    def set_error_color(self):
        msg = self.err_msg.get()
        for k in range(len(self.d_tag)):
            fgcolor = 'white' if self.d_tag[k] in msg else 'black'
            bgcolor = 'red'   if self.d_tag[k] in msg else 'white'
            self.var[k].configure(bg=bgcolor, fg=fgcolor)
        for k in range(len(self.b_tag)):
            fgcolor = 'white' if self.b_tag[k] in msg else 'black'
            bgcolor = 'red'   if self.b_tag[k] in msg else 'white'
            self.lb[k].configure(bg=bgcolor, fg=fgcolor)
            self.ub[k].configure(bg=bgcolor, fg=fgcolor)

    #=========================================================================
    # load_parameters:  Load parameter values from GUI entry fields
    # ------------------------------------------------------------------------
    #  CALLED BY: solve_cb, plot_cb
    #  CALLS:     clear_plot, set_error_color, ConfoundingInterval
    #  VARIABLES:
    #   my_ci = a CI instance constructed by the other values
    #   var   = vector containing values of rho, sy, and sx statistics
    #   lb/ub = arrays of variable lower and upper bounds
    #   msg   = error message returned from bad data
    # ------------------------------------------------------------------------
    #  FUNCTIONS:
    #   process_error = process a user input error
    #=========================================================================
    def load_parameters(self):

        # Sub-function for processing input Entry error
        def process_error(msg,var=''):
            self.err_msg.set('ERROR:  ' + str(msg) + ' ' + var)
            self.err_text.grid()
            self.clear_plot()
            self.out_frame.grid_remove()

        # Retrieve variable values from GUI input fields
        my_ci = []
        var = [None]*len(self.d_var)
        for k in range(len(self.d_var)):
            try:
                var[k] = float(self.d_var[k].get())
            except ValueError as msg:
                process_error(msg,self.d_tag[k])
                self.set_error_color()
                return my_ci

        # Retrieve bound values from GUI input fields
        lb = [None]*len(self.lb_var)
        ub = [None]*len(self.ub_var)
        for k in range(len(self.ub_var)):
            try:
                lb[k] = float(self.lb_var[k].get())
                ub[k] = float(self.ub_var[k].get())
            except ValueError as msg:
                process_error(msg,self.b_tag[k])
                self.set_error_color()
                return my_ci

        # Try to construct a CI instantiation with error trapping
        try:
            my_ci = ConfoundingInterval(var[0],var[1],var[2],lb,ub)
            self.err_msg.set('None')
            self.err_text.grid_remove()
        except ValueError as msg:
            process_error(msg)
        self.set_error_color()
        return my_ci

    #=========================================================================
    # clear_plot:  Clear the current plot
    # ------------------------------------------------------------------------
    #  CALLED BY:  plot_cb, load_problem
    #=========================================================================
    def clear_plot(self):
        if self.canvas != []:
            self.canvas.get_tk_widget().destroy()
            self.canvas = []

    #=========================================================================
    # validate_entry:  Validate an entry in an EntryTable
    # ------------------------------------------------------------------------
    #  CALLED BY:  MainApplication
    #  VARIABLES:
    #   minus  = string set equal to '-' or '', depending on the entry field
    #   action = code for type of user action (1 = insert)
    #   val    = string entered into the Entry field
    #   S      = code for characters in the Entry field
    #=========================================================================
    def validate_entry(self,minus,action,val,S):
        if action != '1':  return True
        if S not in '0123456789.' + minus:  return False
        try:
            float(val)
            return True
        except ValueError:
            return False

    #=========================================================================
    # problem_cb:  Callback function for the Problem popup menu
    # ------------------------------------------------------------------------
    #  CALLED BY:  MainApplication
    #  CALLS:      load_problem
    #=========================================================================
    def problem_cb(self,*args):
        self.load_problem(self.prob.get())

    #=========================================================================
    # reset_cb:  Callback function for resetting GUI to default values
    # ------------------------------------------------------------------------
    #  CALLED BY:  MainApplication
    #  CALLS:      load_problem
    #=========================================================================
    def reset_cb(self):
        self.prob.set(self.prob_choice[0])
        self.alg.set(self.alg_choice[0])
        self.load_problem(self.prob.get())

    #=========================================================================
    # solve_cb:  Callback function for Solve push button
    # ------------------------------------------------------------------------
    #  CALLED BY:  MainApplication
    #  CALLS:      load_parameters, my_ci.optimize
    #  VARIABLES:
    #   my_ci = instantiation of a ConfoundingInterval object
    #=========================================================================
    def solve_cb(self):
        my_ci = self.load_parameters()
        if (my_ci == []): return
        my_ci.optimize(self.alg.get(),'grid',3)
        self.out_frame.grid(row=4,column=0,pady=(5,0))
        self.out_ci[2].configure(text='{:.4f}'.format(my_ci.fx_min))
        self.out_ci[3].configure(text='{:.4f}'.format(my_ci.fx_max))
        for k in range(len(self.min)):
            self.min[k].configure(text='{:.4f}'.format(my_ci.x_min[k]))
            self.max[k].configure(text='{:.4f}'.format(my_ci.x_max[k]))

    #=========================================================================
    # plot_cb:  Callback function for Plot push button
    # ------------------------------------------------------------------------
    #  CALLED BY:  MainApplication
    #  CALLS:      load_parameters, my_ci.get_plot_data, on_press
    #  VARIABLES:
    #   my_ci = instantiation of a ConfoundingInterval object
    #   x     = vector evenly spaced points in (-1,1) for plotting
    #   lo/hi = confounding interval bounds for each value of x
    #   ax    = axes of the figure
    # ------------------------------------------------------------------------
    #  FUNCTIONS:
    #   on_press = copy plot to new window when embedded GUI plot is clicked
    #=========================================================================
    def plot_cb(self):

        # Copies plot to new figure window when embedded plot is clicked
        def on_press(event):
            plt.figure()
            plt.plot(x,lo,x,hi)
            plt.title(self.txt_title)
            plt.ylabel(self.txt_y)
            plt.xlabel(self.txt_x)
            plt.xlim(-1,1)
            plt.legend(self.txt_legend)
            plt.show()      

        self.clear_plot()
        my_ci = self.load_parameters()
        if (my_ci == []): return
        [x,lo,hi] = my_ci.get_plot_data(self.alg.get())
        self.fig = Figure()
        ax = self.fig.add_subplot(111)
        ax.plot(x,lo,x,hi)
        ax.set_title(self.txt_title)
        ax.set_ylabel(self.txt_y)
        ax.set_xlabel(self.txt_x)
        ax.set_xbound(-1,1)
        ax.legend(self.txt_legend)
        self.canvas = FigureCanvasTkAgg(self.fig,self.parent)
        self.canvas.show()
        self.canvas.mpl_connect('button_press_event', on_press)
        self.canvas.get_tk_widget().grid(row=5,column=0)

#=============================================================================
# Main script for launching GUI
#=============================================================================
if __name__ == "__main__":
    gui = tk.Tk()
    gui.title('Compute Confounding Interval')
    gui.option_add('*Font','Helvetica 11')
    gui.option_add('*Button.Font','Helvetica 14 bold')
    gui.option_add('*Button.foreground','white')
    gui.option_add('*Button.width',17)
    MainApplication(gui).grid()
    gui.mainloop()
