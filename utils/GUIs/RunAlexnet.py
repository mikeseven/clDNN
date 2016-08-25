#!/usr/bin/python
import os
import webbrowser
import common
import traceback
from common.gui_common import *

config_file = os.path.join(os.path.dirname(__file__), 'conf.ini')

section = 'Run Alexnet'
build_type_str = 'Build Type'
architecture_type_str = 'Architecture Type'
networks_str = 'Networks'
runs_on_str = 'Run On'
batch_num_str = 'Batch Num'
input_dir_str = 'Input Images Dir'
network_files_dir_str = 'Network Files Dir'
dump_to_file_str = 'Dump Intermediate Results'


class AlexnetDialog(GuiCommon):
    def __init__(self, default_params, master=None):
        GuiCommon.__init__(self, default_params, section, master)
        self.networks = ["alexnet"]
        self.runs_on = ["gpu", "reference"]
        self.batch_num_options = range(1,33)
        self.mainframe()

    def mainframe(self):
        row = 0

        self.add_all_line_text(window=self, text="Alexnet", row=row, font_size=14, padx=10, pady=5, fg='blue')
        row += 1

        frame = Frame(self, bd=2, relief=SUNKEN)
        frame.grid(row=row, column=0, columnspan=2, padx=5, sticky=W+E+N+S)
        frame.grid_columnconfigure(0, minsize=150)
        frame.grid_columnconfigure(1, minsize=200)
        frame.grid_columnconfigure(2, minsize=50)

        frame_row = self.add_general_data(frame, 0)
        frame_row = self.add_separator(frame, 10, frame_row)

        self.build_type = StringVar()
        self.build_type.set(self.default_params.get(build_type_str, "Release"))

        self.add_label(window=frame, text=build_type_str+": ", row=frame_row, column=0, padx=5, sticky=W)
        self.add_radio_button(frame, "Release", self.build_type, "Release", row=frame_row, column=1, sticky=W)
        frame_row += 1
        self.add_radio_button(frame, "Debug", self.build_type, "Debug", row=frame_row, column=1, sticky=W)
        frame_row += 1

        frame_row = self.add_separator(frame, 10, frame_row)

        self.architecture_type = StringVar()
        self.architecture_type.set(self.default_params.get(architecture_type_str, "32"))

        self.add_label(window=frame, text=architecture_type_str+": ", row=frame_row, column=0, padx=5, sticky=W)
        self.add_radio_button(frame, "32", self.architecture_type, "32", row=frame_row, column=1, sticky=W)
        frame_row += 1
        self.add_radio_button(frame, "64", self.architecture_type, "64", row=frame_row, column=1, sticky=W)
        frame_row += 1

        frame_row = self.add_separator(frame, 10, frame_row)

        self.add_label(window=frame, text=dump_to_file_str+": ", row=frame_row, column=0, padx=5, sticky=W)
        frame_row = self.add_check_button(frame, dump_to_file_str, dump_to_file_str, "", frame_row, 0, column=1, sticky=W)
        frame_row = self.add_option_menu(frame, networks_str, self.networks, frame_row)
        frame_row = self.add_option_menu(frame, runs_on_str, self.runs_on, frame_row)
        frame_row = self.add_option_menu(frame, batch_num_str, self.batch_num_options, frame_row)
        frame_row = self.create_dir_widget(frame, input_dir_str, frame_row)
        frame_row = self.create_dir_widget(frame, network_files_dir_str, frame_row)

        frame_row = self.add_separator(frame, 10, frame_row)

        row += 1

        self.final_button = Button(self, text="Run", command=self.run_test)
        self.final_button.grid(row=row, column=0, columnspan=4, padx=5, pady=5, ipadx=30, ipady=5)

        self.bind_all('<Return>', self.run_test)
        self.bind_all('<Escape>', self.handle_esc)

    def run_test(self, event=None):
        self.master.withdraw()
        try:
            build_type          = self.build_type.get()
            architecture_type   = self.architecture_type.get()
            input_dir           = self.option_menus[input_dir_str].get()
            network_files_dir   = self.option_menus[network_files_dir_str].get()
            network             = self.option_menus[networks_str].get()
            runs_on             = self.option_menus[runs_on_str].get()
            batch_num           = self.option_menus[batch_num_str].get()
            dump_to_file        = common.int_to_str_bool(self.check_buttons[dump_to_file_str][1].get())

            new_default_params = {
                input_dir_str: input_dir,
                network_files_dir_str: network_files_dir,
                networks_str: network,
                runs_on_str: runs_on,
                dump_to_file_str: dump_to_file,
                batch_num_str: batch_num,
                build_type_str: build_type,
                architecture_type_str: architecture_type,
            }
            common.ConfigParserWrapper(config_file).write(section, new_default_params)
            exe_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "out", "Windows"+architecture_type, build_type)
            exe_file = os.path.abspath(os.path.join(exe_dir, "examples"+architecture_type+".exe"))
            args = ' --batch={} --input="{}" --model={} --engine={}'.format(batch_num, input_dir, network, runs_on)
            if dump_to_file == "True":
                args += " --dump_hidden_layers"
            os.chdir(network_files_dir)
            os.system(exe_file + args)
            out_file = None
            for x in os.listdir(network_files_dir):
                if x.endswith(".html"):
                    out_file = x
            full_path_out_file = os.path.realpath(os.path.join(network_files_dir, out_file))

            webbrowser.open(full_path_out_file, new=2)
        except:
            print("error")
            traceback.print_exc(file=sys.stdout)

        raw_input("\nPlease press any key to continue . . .")

        self.quit()

if __name__ == '__main__':
    params_dict = common.ConfigParserWrapper(config_file).read(section)
    d = AlexnetDialog(params_dict)
    d.mainloop()