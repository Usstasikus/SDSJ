import os

import psutil
import pwd

def filter_pick(str_list, filter):
    # iterates over list and checks for each entry, whether filter matches
    match = list()
    try:
        match = [l for l in str_list for m in (filter(l),) if m]
    except IndexError:
        return False
    if len(match) == 1:
        return True
    else:
        return False


def send_signal_to_our_processes(filter, sig=0):
    # Sends sig to all processes matching filter
    contacts = list()
    processes = psutil.process_iter()
    this_process = psutil.Process()
    for proc in processes:
        try:
            if proc.pid == this_process.pid:
                continue
            if proc.username() == pwd.getpwuid(os.getuid()).pw_name:
                # = It is our process
                if filter_pick(str_list=proc.cmdline(), filter=filter):
                    proc.send_signal(sig)
                    contacts.append(proc.pid)
        except Exception as e:
            print(e)
    return contacts