import streamlink
import os

def download_video_url(dirpath, filename, url, resolution='best', debug=False):
    '''
    downloads url to a file in the temporary directory
    Args:
        dirpath (str): directory to donwload file to
        filename (str): filename to save as
        url (str): video url to process
    '''
    output_file = os.path.join(dirpath, filename + '.mp4')
    if debug:
        print(output_file)
    try: 
        stream = streamlink.streams(url)[resolution]
    except Exception as e:
        print("An Error has occured: {}".format(e))
        return None

    with open(output_file, "wb") as f, stream.open() as fd:
        while True:
            data = fd.read(1024)
            if not data:
                break
            f.write(data)
    print("{} video has been downloaded".format(filename))
    return output_file

def url_to_stream(url, resolution='720p60'):
    '''
    Creates stream from url for processes that don't necessarily need a file download.
    Args:
        url: url for video (str)
    Returns:
        stream: raw data stream.

    Usage:
        with open(stream) as s:
            do_stuff(s)
    '''
    try: 
        stream = streamlink.streams(url)[resolution]
    except Exception as e:
        print("An Error has occured: {}".format(e))
        return None
    return stream