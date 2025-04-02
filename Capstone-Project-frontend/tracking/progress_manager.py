from multiprocessing import Manager

manager = Manager()
_progress_value = manager.Value('i', 0)
_final_csv = manager.Value('s', '')

def get_shared_progress():
    return _progress_value, _final_csv
