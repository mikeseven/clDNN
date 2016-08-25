#!/usr/bin/python
from __future__ import print_function

# TODO: THIS IS A MESS... (SORT IT OUT)

import errno
import re
import logging
import os
import shutil
import stat
import subprocess
import platform
import threading
import Queue
from argparse import ArgumentTypeError

import settings

users = {'ekfir': 'eli.kfir',
         'emeiri': 'etay.meiri',
         'hlaty': 'hila.laty',
         'jsubag': 'jacob.subag',
         'mkislev': 'maayan.kislev',
         'rrichman': 'reuven.richman',
         'tbaron': 'tomer.bar.on',
         'tsurazhs': 'tatiana.surazhsky',
         'usarel': 'uzi.sarel',
         'mdejaegh' : 'matthias.dejaegher',
         'lab_rastydaily': 'vpg.gsdv.3d.haifa'}

ERROR_CODE_FAILED = -1
########################################################################################################################
def dump_args(func, print_cwd=False):
    """
    This decorator dumps out the arguments passed to a function before calling it

    Source: https://wiki.python.org/moin/PythonDecoratorLibrary#Easy_Dump_of_Function_Arguments
    """

    arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
    fn = func.func_code.co_filename
    f_name = func.func_name

    def echo_func(*args, **kwargs):
        echo_func.times_called += 1

        logging.debug(('{}, {} (call #{}):\n'
                       '\t{}\n'.format(os.path.basename(fn),
                       f_name,
                       echo_func.times_called,
                       ', '.join('%s=%r' % entry for entry in zip(arg_names, args) + kwargs.items()))))
        # if print_cwd:
        #     print('\tcwd:\t\t{}\n'.format(os.getcwd()))

        return func(*args, **kwargs)

    echo_func.times_called = 0
    return echo_func


########################################################################################################################
def split_to_lines_str(text):
    tabs_expended = text.expandtabs(4)
    lines_split = tabs_expended.splitlines()
    return [line.strip() for line in lines_split]


########################################################################################################################
def split_to_lines(filename):
    logging.debug(('\n'
                   '#2      (split_to_lines)\n'
                   '    cwd     {}\n'
                   '    file    {}\n').format(os.getcwd(),
                                              filename))

    with open(filename, 'r') as f:
        # todo: can actually yield form here? not need to read the entire file?
        # for line in f:
        #   yield line
        return split_to_lines_str(f.read())


########################################################################################################################
@dump_args
def write_to_file(message, filename):
    logging.debug(('\n'
                   '#3      (write_to_file)\n'
                   '    cwd     {}\n'
                   '    msg     {}\n'
                   '    fn      {}\n').format(os.getcwd(),
                                              message,
                                              filename))

    with open(filename, 'a') as log:
        log.write(message)


########################################################################################################################
@dump_args
def run_cmd(cmd, output_file=None):
    if output_file:
        cmd += ' >> ' + output_file + ' 2>&1'

    print("Running command: " + cmd)

    return os.system(cmd)


########################################################################################################################
def get_formatted_time(total_time):
    minutes = int(total_time / 60)
    seconds = int(total_time % 60)
    return float(minutes + float(seconds) / 100)


########################################################################################################################
def get_base_hash():
    try:
        out, _ = subprocess.Popen(['git', 'rev-list', 'origin/{}..HEAD'.format(settings.get_branch_name())], stdout=subprocess.PIPE).communicate()
        out = out.strip()
        if out == '':
            num_of_commits = 0
        else:
            lines = out.split('\n')
            num_of_commits = len(lines)

        out, _ = subprocess.Popen(['git', 'rev-list', '--max-count=' + str(num_of_commits + 1), 'HEAD'], stdout=subprocess.PIPE).communicate()
        return out.strip().split('\n')[-1].rstrip()
    except:
        # fall-back for Etay who has problem with subprocess class
        print('Fall back path')
        temp_filename = 'temp.txt'

        run_cmd('git rev-list origin/' + settings.get_branch_name() + '..HEAD > ' + temp_filename)

        logging.debug(('\n'
                       '#4      (get_base_hash)\n'
                       '    cwd     {}\n'
                       '    temp_fn {}\n').format(os.getcwd(),
                                                  temp_filename))

        # not terribly efficient, could avoid storing the entire file in meme here but this is a one time thing and all...
        with open(temp_filename) as f:
            num_of_commits = len(f.readlines())

        run_cmd('git rev-list --max-count=' + str(num_of_commits + 1) + ' HEAD > ' + temp_filename)

        logging.debug(('\n'
                       '#5      (get_base_hash)\n'
                       '    cwd     {}\n'
                       '    tmp_fn  {}\n').format(os.getcwd(),
                                                  temp_filename))

        with open(temp_filename) as f:
            lines = f.readlines()

        os.remove(temp_filename)

        return lines[-1].rstrip()


########################################################################################################################
def create_patch_file(temp_dir, branch, ignore_uncommited_changes):
    patch_filename_and_location = os.path.join(temp_dir, 'Patchfile.patch')

    if os.path.exists(patch_filename_and_location):
        os.remove(patch_filename_and_location)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    ret = 1
    if not ignore_uncommited_changes:
        ret = run_cmd('git commit -a -m jenkins1')

    cmd = ['git', 'log', '--reverse', '--pretty=format:%H', '-m', '--first-parent', '{}..HEAD'.format(branch)]
    try:
        out, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()
    except:
        temp_filename = 'temp.txt'
        run_cmd(' '.join(cmd) + ' > ' + temp_filename)
        with open(temp_filename) as f:
            out = f.read()
        os.remove(temp_filename)
    print(out)
    #run_cmd('git log -p --reverse --pretty=email --stat -m --first-parent {}..HEAD > {}'.format(branch, patch_filename_and_location))
    #run_cmd('git format-patch ' + branch + ' --stdout > ' + patch_filename_and_location)
    for line in out.split('\n'):
        sha = line.strip()
        if sha:
            run_cmd('git format-patch -1 {} --stdout >> {}'.format(sha, patch_filename_and_location))
    if ret == 0:
        run_cmd('git reset --soft \"HEAD^\"')
    if not os.path.exists(patch_filename_and_location) or os.path.getsize(patch_filename_and_location) == 0:
        safe_remove(patch_filename_and_location)
        print("Error - empty Patch file")
        return None
    return patch_filename_and_location

########################################################################################################################
def check_local_changes():
    branch = get_base_hash()
    ret = run_cmd('git commit -a -m jenkins1')
    out, err = subprocess.Popen('git format-patch ' + branch + ' --stdout', shell=True, stdout=subprocess.PIPE).communicate()
    if ret == 0:
        run_cmd('git reset --soft \"HEAD^\"')
    return out != ""


########################################################################################################################
@dump_args
def copy_specified_files_recursive_pair(source_destination, extension, max_depth=9999):
    def copy_specified_files_recursive(s, d, depth):
        if depth < 0:
            return
        for filename in os.listdir(s):
            src = os.path.join(s, filename)
            dst = os.path.join(d, filename)
            if os.path.isdir(src):
                copy_specified_files_recursive(src, dst, depth-1)
            #elif instead? dirs can have the "ext"
            if filename.endswith(extension):
                if not os.path.exists(d):
                    os.makedirs(d)
                try:
                    shutil.copy(src, dst)
                except IOError as e:
                    print(e)
                    exit(ERROR_CODE_FAILED)

    # print('>>>>>', source, '-----', destination, '<<<<')
    return copy_specified_files_recursive(*source_destination, depth=max_depth)


########################################################################################################################
def get_source_target_pair(source_root, target_root, relative_path):
    source = os.path.join(source_root, relative_path)
    target = os.path.join(target_root, relative_path)

    if not os.path.isdir(target):
        os.makedirs(target)

    return source, target


########################################################################################################################
@dump_args
def hard_delete(path):
    def handle_remove_readonly(func, path, exc):
        exc_value = exc[1]
        if exc_value.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
            return func(path)
        else:
            print("func=<",func, "> path=<", path, "> exc =<", exc, "> errno <", exc_value.errno, ">")
            raise Exception

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=False, onerror=handle_remove_readonly)
    elif os.path.isfile(path):
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        os.remove(path)

########################################################################################################################
@dump_args
def renew_dir(dir_path):
    hard_delete(dir_path)
    os.makedirs(dir_path)


########################################################################################################################
def exit_error(msg):
    print('Error - ' + msg)
    exit(ERROR_CODE_FAILED)


########################################################################################################################
def handle_error(ret, msg):
    if ret:
        exit_error(msg)


########################################################################################################################
def disable_debugger():
    reg_file = os.path.join(os.path.dirname(__file__), '../WindowsUseful/DisableDebbuger.reg')
    return run_cmd("regedt32.exe /s " + reg_file)


########################################################################################################################
def enable_debugger():
    reg_file = os.path.join(os.path.dirname(__file__), '../WindowsUseful/EnableDebbuger.reg')
    return run_cmd("regedt32.exe /s " + reg_file)


########################################################################################################################
@dump_args
def safe_copy(src, dst):
    try:
        shutil.copy(src, dst)
    except (IOError, OSError) as e:
        logging.warning("Couldn't copy {} to {}, got\n{}".format(src, dst, e))

########################################################################################################################
@dump_args
def safe_remove(dst):
    try:
        os.remove(dst)
    except (IOError, OSError) as e:
        logging.warning("Couldn't remove {}, got\n{}".format(dst, e))

########################################################################################################################
@dump_args
def safe_move(src, dst):
    try:
        shutil.move(src, dst)
    except (IOError, OSError) as e:
        print("\nCouldn't copy {} to {}, got\n{}".format(src, dst, e))


########################################################################################################################
def get_user():
    try:
        if platform.system() == 'Windows':
            cmd = "net user {} /domain".format(os.environ['username'])
            out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()
            prefix = "Full Name"
            short_email = ""
            for line in out.split('\n'):
                line = line.strip()
                if line.startswith(prefix):
                    name = line[len(prefix):].strip()
                    splitted = name.split(',')
                    short_email = splitted[0].strip().replace(' ', '.')
                    if len(splitted) == 2:
                        short_email = splitted[1].strip().replace(' ', '.') + '.' + short_email
                    short_email = short_email.lower()
                    break
            if short_email == '':
                short_email = users[os.getenv('username')]
            return short_email
        else:
            email = os.getenv('EMAIL')
            intel_postfix = '@intel.com'
            if email.endswith(intel_postfix):
                short_email = email[:-len(intel_postfix)]
                return short_email
    except:
        return users[os.getenv('username')]

########################################################################################################################
def int_to_str_bool(b):
    return 'False' if b == 0 else 'True'

########################################################################################################################
def str_to_int_bool(b):
    return 0 if b == 'False' else 1

########################################################################################################################
class FdReaderNoneBlocking:
    def __init__(self, fd):
        self.fd = fd
        self.queue = Queue.Queue()
        self.thread = threading.Thread(target = self.thread_main)
        self.thread.setDaemon(True)
        self.thread.start()

    def thread_main(self):
            while True:
                line = self.fd.readline()
                if line:
                    self.queue.put(line)
                else:
                    break

    def readline(self, timeout = 0.1):
        try:
            block_read = timeout is not None
            return self.queue.get(block = block_read, timeout = timeout)
        except Queue.Empty:
            return None

########################################################################################################################
def run_cmd_tee(cmd, file_name, sentinel=b''):
    print("run_cmd_tee({}, {})\n".format(cmd, file_name))
    with open(file_name, 'w') as fd:
        p = subprocess.Popen(cmd + ' 2>&1', shell=True, stdout=subprocess.PIPE)
        nbsr = FdReaderNoneBlocking(p.stdout)
        while True:
            line = nbsr.readline(0.1)
            #line = p.stdout.readline()
            if not line:
                res = p.poll()
                if res is not None:
                    break
            else:
                line = line.rstrip().replace('\r', '')
                print(line)
                fd.write(line + '\n')
        logging.info(str(nbsr.thread.isAlive()))
        return p.returncode

########################################################################################################################
def parse_multi_range_param(p, parse_infinite = True, max_val=100):
    l = []
    if p is not None and isinstance(p, str):
        for num in p.split(','):
            m = re.match(r'(\d+)(?:-(\d+)?)?$', num)
            if m is not None:
                start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                else:
                    if parse_infinite and '-' in num:
                        end = max_val
                    else:
                        end = start
                l += list(range(start, end+1))
            else:
                raise ArgumentTypeError(num)
    return l


########################################################################################################################
class CommitLog(object):
    def __init__(self, data = list()):
        if len(data) == 5:
            self.sha = data[0].strip()
            self.author = data[1].strip()
            self.author_email = data[2].strip()
            self.date = data[3].strip()
            self.subject = data[4].strip()
        # else:
        #     print(data)

    def __repr__(self):
        pretty_format = \
            '\n' \
            'Commit:        {}\n' \
            'Author:        {}\n' \
            'Author Email:  {}\n' \
            'Date:          {}\n' \
            'Subject:       {}\n'.format(
                self.sha,
                self.author,
                self.author_email,
                self.date,
                self.subject
            )
        return pretty_format

    @staticmethod
    def parse_file(file_name):
        res_list = []
        try:
            with open(file_name) as f:
                lines = f.readlines()
                for i in range(0, len(lines), 6):
                    data = [d.split(':',1)[1] for d in lines[i+1:i+6]]
                    c = CommitLog(data)
                    res_list.append(c)
        except:
            pass
        return res_list

########################################################################################################################
def get_revisions_log(revisions, git_dir):
    logs = []
    if len(revisions) == 2 and revisions[0] != revisions[1]:
        try:
            curr_dir = os.getcwd()
            os.chdir(git_dir)
            format = \
                '%H%n' \
                '%aN%n' \
                '%aE%n' \
                '%aD%n' \
                '%s'
            cmd = ['git', 'log', '--reverse', '--pretty=format:'+format, '-m', '--first-parent', revisions[1] +'..' + revisions[0]]
            out, _ = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()
            out_lines = out.split('\n')

            for i in range(0, len(out_lines), 5):
                logs.append(CommitLog(out_lines[i:i+5]))
            os.chdir(curr_dir)
        except:
            pass
    return logs

########################################################################################################################
class PropHandler(object):
    def __init__(self, dir):
        prop_file_name = os.path.abspath(dir)
        prop = {}
        with open(prop_file_name) as file:
            current_var = None
            for line in file:
                current_str = line
                if '=' in current_str:
                    current_var, current_str = current_str.split('=')
                    prop[current_var] = []
                if '\\' in current_str:
                    current_str = current_str.split('\\')[0]
                current_str = current_str.strip()
                if current_str.endswith(','):
                    current_str = current_str[:len(current_str)-1]
                if current_var and current_str:
                    prop[current_var] += current_str.split(',')
        self.prop = prop

    ####################################################################################################################
    def get_category_list(self):
        return [x for x in self.prop]

    ####################################################################################################################
    def get_category(self, category):
        for test in self.prop[category]:
            base_test = test.split(' ')[0]
            text = base_test + ' '

            logging.info(test + ' ... ' + text)
            yield [test, text]