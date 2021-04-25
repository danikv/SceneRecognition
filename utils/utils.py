def get_number_from_filename(filename):
    return int(filename.split('_')[-1].split('.')[0])

def parse_time(string_time):
    splitted_string = string_time.split(':')
    minutes = int(splitted_string[0])
    seconds = int(splitted_string[1])
    return minutes, seconds