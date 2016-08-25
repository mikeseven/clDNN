#!/usr/bin/python
import os
import tkFont
import tkMessageBox
import tkFileDialog
from Tkinter import *
import __init__ as common

ignore_uncommited_changes_str = 'Ignore Uncommited Changes'

class GuiCommon(Frame):
    def __init__(self, default_params, title, master=None):
        Frame.__init__(self, master)
        self.grid(sticky=N+S+E+W)
        self.master.title(title)
        self.labels = {}
        self.option_menus = {}
        self.option_menus_widgets = {}
        self.check_buttons = {}
        self.default_params = default_params

    def add_all_line_text(self, window, row, text, font_size, weight="bold", bg=None, fg=None, **options):
        if not bg:
            bg = self.master.cget('bg')
        if not fg:
            fg = 'black'
        font = tkFont.Font(family="Helvetica", size=font_size, weight=weight)
        label = Label(window, text=text, font=font, bg=bg, fg=fg)
        label.grid(row=row, column=0, columnspan=10, **options)

    def add_check_button(self, window, name, val_key, text, row, default, **options):
        var = IntVar()
        var.set(common.str_to_int_bool(self.default_params.get(val_key, common.int_to_str_bool(default))))
        check_button = Checkbutton(window, text=text, variable=var)
        check_button.grid(row=row, **options)
        self.check_buttons[name] = (check_button, var)
        return row + 1

    @staticmethod
    def add_label(window, text, **options):
        label = Label(window, text=text)
        label.grid(**options)
        return label

    def add_radio_button(self, window, text, var, val, **options):
        r = Radiobutton(window, text=text, variable=var, value=val)
        r.grid(**options)
        return r

    @staticmethod
    def add_separator(window, height, row):
        line = Canvas(window, height=height, width=1)
        line.grid(row=row, column=0, columnspan=2)
        return row+1

    def create_dir_widget(self, frame, param_str, frame_row):
        self.add_label(window=frame, text=param_str+": ", row=frame_row, column=0, padx=5, sticky=W)
        self.option_menus[param_str] = StringVar()
        self.option_menus[param_str] = StringVar()
        self.option_menus[param_str].set(self.default_params.get(param_str, ''))
        self.input_dir_widget = Entry(frame, textvariable=self.option_menus[param_str])
        self.input_dir_widget.grid(row=frame_row, column=1, columnspan=1, padx=2, sticky=W+E)
        Button(frame, text="Browse", command=lambda e=None: self.browse(param_str)).grid(row=frame_row, column=2, padx=5, pady=5, ipadx=30)
        return  frame_row + 1

    def add_option_menu(self, window, param_str, vals, row):
        self.option_menus[param_str] = StringVar()
        self.option_menus[param_str].set(self.default_params.get(param_str, vals[0]))

        self.add_label(window=window, text=param_str+": ", row=row, column=0, padx=5, sticky=W)

        w = OptionMenu(window, self.option_menus[param_str], *vals)
        w.config(bg='white', width=50, anchor='w')
        w.grid(row=row, column=1, sticky=W)

        self.option_menus_widgets[param_str] = w

        return row + 1

    def add_general_data(self, frame, frame_row):
        self.add_all_line_text(frame, frame_row, "Hello " + common.get_user(), 10, padx=5, sticky=W)
        frame_row += 1

        self.add_label(window=frame, text="Git Repo: ", row=frame_row, column=0, padx=5, sticky=W)
        self.add_label(window=frame, text=common.get_git_repo(), row=frame_row, column=1, padx=2, sticky=W)
        frame_row += 1

        self.add_label(window=frame, text="Base Branch: ", row=frame_row, column=0, padx=5, sticky=W)
        self.add_label(window=frame, text=common.get_branch_name(), row=frame_row, column=1, padx=2, sticky=W)
        frame_row += 1
        return frame_row

    def browse(self, param_str):
        app = tkFileDialog.askdirectory(title=param_str)
        self.option_menus[param_str].set(app)

    def handle_esc(self, event=None):
        self.master.withdraw()
        self.quit()

########################################################################################################################
def get_patch_file(ignore_no_changes, temp_dir, branch, ignore_uncommited_changes):
    patch_file = common.create_patch_file(temp_dir, branch, ignore_uncommited_changes)
    if not patch_file:
        if ignore_no_changes:
            patch_file = os.path.join(temp_dir, 'empty.patch')
            f = open(patch_file, 'w')
            f.close()
        else:
            print("Error - no changes to check")
            print("Exiting")
            raw_input()
            exit(1)

    return patch_file


########################################################################################################################
def run_simple_test(build_driver, params):
    pass

########################################################################################################################
def run_medium_test(build_driver, params):
    pass

########################################################################################################################
def run_build_test(params):
    pass
