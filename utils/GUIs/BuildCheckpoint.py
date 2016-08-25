#!/usr/bin/python
import common
from common.gui_common import *

config_file = os.path.join(os.path.dirname(__file__), 'conf.ini')
section = 'Build Checkpoint'

class MediumCheckpointDialog(GuiCommon):
    def __init__(self, default_params, master=None):
        GuiCommon.__init__(self, default_params, section, master)
        self.mainframe()

    def get_key_from_test(self, group, test):
        return '{} {}'.format(group, test)

    def add_tests_to_frame(self, frame, category, filter, title, start_row):
        frame_row = start_row
        self.add_label(window=frame, text=title+": ", row=frame_row, column=0, padx=5, sticky=W)

        key = category + '@' + filter
        self.check_buttons[key] = {}
        for test, text in self.prop.get_category(category):
            if filter in test:
                val_key = self.get_key_from_test(key, test)
                self.add_check_button(frame, test, val_key, test, frame_row, 0, column=1, sticky=W)
                frame_row += 1

        return frame_row

    def mainframe(self):
        self.prop = common.PropHandler(os.path.dirname(__file__) + "/Props/Prop.properties")

        row = 0

        self.add_all_line_text(window=self, text=section, row=row, font_size=14, padx=10, pady=5, fg='blue')
        row += 1

        if not common.check_local_changes():
            self.add_all_line_text(window=self, text="Warning - no local changes", row=row, font_size=10, fg='red', padx=5, pady=5)
            row += 1


        frame = Frame(self, bd=2, relief=SUNKEN)
        frame.grid(row=row, column=0, columnspan=3, padx=5, pady=10)
        frame.grid_columnconfigure(0, minsize=150)
        frame.grid_columnconfigure(1, minsize=100)

        self.add_general_data(frame, 0)

        row += 1

        frame = Frame(self, bd=2, relief=SUNKEN)
        frame.grid(row=row, column=0, padx=5, sticky=W+E+N+S)
        frame.grid_columnconfigure(0, minsize=70)
        frame.grid_columnconfigure(1, minsize=200)

        frame_row = 0
        self.add_all_line_text(window=frame, text="Windows", row=frame_row, font_size=10, padx=10, pady=5, fg='blue')
        frame_row += 1
        frame_row = self.add_tests_to_frame(frame, 'WindowsBuild', '', 'Gen9', frame_row)

        frame = Frame(self, bd=2, relief=SUNKEN)
        frame.grid(row=row, column=1, padx=5, sticky=W+E+N+S)
        frame.grid_columnconfigure(0, minsize=70)
        frame.grid_columnconfigure(1, minsize=200)

        frame_row = 0
        self.add_all_line_text(window=frame, text="Linux", row=frame_row, font_size=10, padx=10, pady=5, fg='blue')
        frame_row += 1
        frame_row = self.add_tests_to_frame(frame, 'LinuxBuild', '', 'Gen9', frame_row)

        row += 1
        row = self.add_separator(self, 10, row)

        row += 1

        frame = Frame(self, bd=2)
        frame.grid(row=row, column=0, columnspan=3, padx=5, pady=10)
        frame.grid_columnconfigure(0, minsize=70)
        frame.grid_columnconfigure(1, minsize=30)
        frame.grid_columnconfigure(2, minsize=100)

        frame_row = 0
        self.add_label(window=frame, text="Notes: ", row=frame_row, column=1, sticky=E)
        self.notes_var = StringVar()
        self.notes = Entry(frame, textvariable=self.notes_var)
        self.notes.grid(row=frame_row, column=2, padx=2, sticky=W)

        frame_row += 1
        cb_str = ignore_uncommited_changes_str
        frame_row = self.add_check_button(frame, cb_str, cb_str, cb_str, frame_row, 0, column=1, columnspan=2, sticky=W)

        row += 1

        self.final_button = Button(self, text="Run", command=self.run_test)
        self.final_button.grid(row=row, column=0, columnspan=4, padx=5, pady=5, ipadx=30, ipady=5)

        self.bind_all('<Return>', self.run_test)
        self.bind_all('<Escape>', self.handle_esc)

    def run_test(self, event=None):
        conf_params = {}
        params = {}
        for key, val in self.check_buttons.items():
            tests_str = ""
            for key1, val1 in val.items():
                if val1[1].get() == 1:
                    val_key = self.get_key_from_test(key, key1)
                    conf_params[val_key] = common.int_to_str_bool(val1[1].get())
                    tests_str += key1 + ","
            param_key = key.split('@')[0].strip()
            if not param_key in params:
                params[param_key] = ''
            if tests_str:
                params[param_key] += tests_str
        for key, val in params.items():
            params[key] = '"' + val[:-1] + '"'
        params['IgnoreNoChanges'] = True
        params['Notes'] = self.notes_var.get()
        params[ignore_uncommited_changes_str] = common.int_to_str_bool(self.check_buttons['misc'][ignore_uncommited_changes_str][1].get())
        common.ConfigParserWrapper(config_file).write(section, conf_params)
        run_build_test(params)
        self.master.withdraw()

        tkMessageBox.showinfo(message="Go get Cup-Of-Coffee")
        self.quit()

if __name__ == '__main__':
    params = common.ConfigParserWrapper(config_file).read(section)
    d = MediumCheckpointDialog(params)
    d.mainloop()