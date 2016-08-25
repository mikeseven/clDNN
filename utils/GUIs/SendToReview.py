#!/usr/bin/python
import common
from common.gui_common import *

config_file = os.path.join(os.path.dirname(__file__), 'conf.ini')
section = 'Send To Review'
review_str = 'Code Review'
review_options = ['Add New Review', 'Choose Existing Review']
use_local_build_str = 'Use Local Build'
title_str = "Title"


class SendToReviewDialog(GuiCommon):
    def __init__(self, default_params, master=None):
        GuiCommon.__init__(self, default_params, section, master)
        self.mainframe()

    def mainframe(self):
        row = 0

        self.add_all_line_text(window=self, text=section, row=row, font_size=14, padx=10, pady=5, fg='blue')
        row += 1

        if not common.check_local_changes():
            self.add_all_line_text(window=self, text="Warning - no local changes", row=row, font_size=10, fg='red', padx=5, pady=5)
            row += 1

        frame = Frame(self, bd=2, relief=SUNKEN)
        frame.grid(row=row, column=0, columnspan=2, padx=5, sticky=W+E+N+S)
        frame.grid_columnconfigure(0, minsize=150)
        frame.grid_columnconfigure(1, minsize=100)

        frame_row = self.add_general_data(frame, 0)
        frame_row = self.add_separator(frame, 10, frame_row)

        self.test = StringVar()
        self.test.set(self.default_params.get(review_str, review_options[0]))

        self.add_label(window=frame, text=review_str + ": ", row=frame_row, column=0, padx=5, sticky=W)
        for review in review_options:
            self.add_radio_button(frame, review, self.test, review, row=frame_row, column=1, sticky=W)
            frame_row += 1

        self.add_label(window=frame, text=ignore_uncommited_changes_str + ": ", row=frame_row, column=0, padx=5, sticky=W)
        cb_str = ignore_uncommited_changes_str
        frame_row = self.add_check_button(frame, cb_str, cb_str, "", frame_row, 0, column=1, sticky=W)

        self.add_label(window=frame, text=title_str+": ", row=frame_row, column=0, padx=5, sticky=W)
        self.title_var = StringVar()
        self.notes = Entry(frame, textvariable=self.title_var)
        self.notes.grid(row=frame_row, column=1, padx=2, sticky=W)

        row += 1

        self.final_button = Button(self, text="Run", command=self.run_test)
        self.final_button.grid(row=row, column=0, columnspan=4, padx=5, pady=5, ipadx=30, ipady=5)

        self.bind_all('<Return>', self.run_test)
        self.bind_all('<Escape>', self.handle_esc)

    def run_test(self, event=None):
        params = {}
        params[review_str] = self.test.get()
        params[ignore_uncommited_changes_str] = common.int_to_str_bool(self.check_buttons[ignore_uncommited_changes_str][1].get())
        common.ConfigParserWrapper(config_file).write(section, params)

        params[title_str] = self.title_var.get()
        params['IgnoreNoChanges'] = False
        #send_to_review(params)
        self.master.withdraw()

        tkMessageBox.showinfo(message="Go get a cup of coffee!")
        self.quit()


if __name__ == '__main__':
    params = common.ConfigParserWrapper(config_file).read(section)
    d = SendToReviewDialog(params)
    d.mainloop()


