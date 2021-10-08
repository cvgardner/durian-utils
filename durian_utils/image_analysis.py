import pandas as pd
import numpy as np
import json
import imutils
import cv2
import os
import time
import string
import re
import math
import pytesseract
import tqdm
import argparse
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import datetime

def preprocess(im):
    '''
    preprocess image with guassian bulr and canny edge detect to help find features
    '''
    blur = cv2.GaussianBlur(im, (5,5) ,1)
    canny = cv2.Canny(blur, 50,500)
    blur2 = cv2.GaussianBlur(canny, (51,51) ,1)
    return blur2

def scaling_match_template(im1, template, threshold=0.3, scale=[0.3, 1.5, 5]):
    '''
    Scaling template to find best matching. currently uses 0.3, 0.6,  0.9 for speed.
    Recommended to preprocess im and template before this function for best results.
    args:
        im1: input image array. ex: loaded from cv2.imread()
        template: template image array. ex: loaded from cv2.imread()
        threshold: threshold for positive match found. Changes sensitivity of detection
        scale (list): used to create linspace for scaling [lowest scaling, higest scaling, number of steps]
    '''
    im = im1.copy()
    tw, th = template.shape[::-1]    
    found = None
    #loop over template sizes
    for scale in np.linspace(scale[0], scale[1], scale[2])[::-1]:
        resized = imutils.resize(im1, width = int(im1.shape[1] / scale))
        r = im1.shape[1] / float(resized.shape[1]) #resize ration

        #break loop if image is smaller than template
        if resized.shape[0] < th or resized.shape[1] < tw:
            break

        #matching
        result = cv2.matchTemplate(resized,template,cv2.TM_CCOEFF_NORMED)
        (_, MaxVal, _, MaxLoc) = cv2.minMaxLoc(result)
        if found is None or found[1] < MaxVal:
            found = (MaxLoc, MaxVal, r)
            #print(scale, found[1], MaxVal)
    if found is not None and found[1] > threshold:
        return found
    else:
        return False

def cleanString(s, inplace=False):
    '''cleans up string from punctuation and newlines'''
    return s.replace('\n','').translate(str.maketrans('','', string.punctuation))

def cropIm(im, box):
    '''
    Crops image using bounding box based on ratios.
    Args: 
        box: [startX, endX, startY, endY]
    Returns:
        temp: cropped image
    '''
    startX, endX, startY, endY = box
    h,w = im.shape
    temp = im[int(h*startY):int(h*endY), int(w*startX):int(w*endX)].copy()

    return temp

def processIm(im, box):
    '''
    Uses Pytesseract to extract text from an image and a selection bounding box

    Args: 
        im: image to process
        box: [startX, endX, startY, endY]
    Returns:
        text: text extracted by pytesseract
    '''
    temp = cropIm(im, box)

    ret, temp = cv2.threshold(temp, 125, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(temp)
    #print(text)
    text = cleanString(text)

    return text

def processReplayIm(im, winIm, loseIm, hero_arts_true):
    '''
    Processes a Replay image using winIm and loseIm as templates
    Args:
        im: cv2 image (numpy array) to process
        winIm: cv2/numpy array image for win template matching
        loseIm: cv2/numpy array image for lose template matching
        hero_arts_true: Dictionary of Hero Arts for fuzzywuzzy comparison
    Returns:
        output: Dictionary of the form
                {
                    hero_art1: str
                    hero_art2: str
                    winner: str
                    timestamp: datetime object
                }
    '''
    #threshold image
    ret, thres = cv2.threshold(im, 125, 255, cv2.THRESH_BINARY)
    img = thres.copy()

    #Get Win locations
    print("Template Matching Wins")
    template = winIm
    result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
    tW, tH = template.shape[::-1]
    matchThreshold = 100000000
    (yCoordsWin, xCoordsWin) = np.where(result >= matchThreshold)

    #Get Lose locations
    print('Template Matching Losses')
    template = loseIm
    result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
    tW, tH = template.shape[::-1]
    matchThreshold = 100000000
    (yCoordsLose, xCoordsLose) = np.where(result >= matchThreshold)

    #flip yCoords for processing
    print("Processing Coordinates")
    yCoordsWin = yCoordsWin[::-1]
    yCoordsLose = yCoordsLose[::-1]

    #binning for expected replay locations
    bins = [0,300, 500, 700, 1000]
    binnedWin = np.digitize(yCoordsWin, bins)
    binnedLose = np.digitize(yCoordsLose, bins)

    #get pruned yCoords
    prev = 5
    yIndexWin = []
    for i, x in enumerate(binnedWin):
        if x <  prev:
            prev=x
            yIndexWin.append(i)
    print(yIndexWin)

    prev = 5
    yIndexLose = []
    for i, x in enumerate(binnedLose):
        if x <  prev:
            prev=x
            yIndexLose.append(i)
    print(yIndexLose)

    #create output
    print("Extracting Hero Arts")
    clone = img.copy()
    h,w = clone.shape

    output = []

    #loop over coordinates
    for i in yIndexWin:
        y = yCoordsWin[i]
        cv2.rectangle(clone, (int(w*0.15), y+50), (int(w*0.85), y + 150),(255, 0, 0), 3)
        #print(pytesseract.image_to_string(clone[y:y+150,:]).replace('\n',''))
        #print(pytesseract.image_to_string(clone[y+50:y+150,:], config='-c preserve_interword_spaces=1').split('\n'))
        text_components = [s for s in pytesseract.image_to_string(clone[y+50:y+150,:], config='-c preserve_interword_spaces=1').split('\n') if 'Replays' in s or 'Share' in s]
        if len(text_components) > 0: #skip if it didn't process correctly
            text_components = text_components[0].split('  ')
        else:
            continue
        hero_arts = list(map(lambda x: process.extractOne(x, hero_arts_true.keys())[0], [s for s in text_components if re.sub(r'\W+', '', s)][:2]))
        print(list(hero_arts))
        data = {
            "hero_art1": hero_arts[0],
            "hero_art2": hero_arts[1],
            "winner": hero_arts[0],
            "timestamp": datetime.datetime.utcnow()
        }
        output.append(data)
        
    for i in yIndexLose:
        y = yCoordsLose[i]
        cv2.rectangle(clone, (int(w*0.15), y+50), (int(w*0.85), y + 150),(255, 0, 0), 3)
        #print(pytesseract.image_to_string(clone[y+50:y+150,:], config='-c preserve_interword_spaces=1').split('\n'))
        text_components = [s for s in pytesseract.image_to_string(clone[y+50:y+150,:], config='-c preserve_interword_spaces=1').split('\n') if 'Replays' in s or 'Share' in s]
        if len(text_components) > 0: #skip if it didn't process correctly
            text_components = text_components[0].split('  ')
        else:
            continue
        hero_arts = list(map(lambda x: process.extractOne(x, hero_arts_true.keys())[0], [s for s in text_components if re.sub(r'\W+', '', s)][:2]))
        print(list(hero_arts))
        data = {
            "hero_art1": hero_arts[0],
            "hero_art2": hero_arts[1],
            "winner": hero_arts[1],
            "timestamp": datetime.datetime.utcnow()
        }
        output.append(data)

    return output

        


def processVideo(video):
    """
    Process frames of video to grab match data. Uses 1 frame every sixty seconds.
    Args:
        video: path to video file to process
    Returns:
        output: json out put of the form 
                { 
                    "hero_art": string,
                    "ranking_points": integer,
                    "timestamp": datetime
                }
    """
    cap = cv2.VideoCapture(video)
    frame_count = 0 #start frames

    with open("config/box_config.json") as f:
        boxes = json.load(f)
    with open("config/hero_arts.json") as f:
        hero_arts = json.load(f)
    latest_HA = ""
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        #use different bounding boxes depending on image ratio
        ratio = str(frame.shape[1] / frame.shape[0])[:3]
        herobox = boxes[ratio]['hero_art']
        pointsbox = boxes[ratio]['ranking_points']

        hero_art = processIm(frame, herobox)
        ranking_points = processIm(frame, pointsbox)
        hero_match = process.extractOne(hero_art, hero_arts.keys())[0]

        # Testing
        # print("Frame #: {}, Hero Art: {}, Matched Art: {}".format(frame_count, hero_art, hero_match))
        
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
       
        # If a Hero Art Match is Found Return



        #move video ahead 1 second = 60 frames
        jump = 60
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count+jump)
        frame_count += 60

        if frame_count > 3*60*60: # if longer than 3 min
            return None

        if latest_HA != "Unknown" and latest_HA == hero_match:
            cap.release()
            cv2.destroyAllWindows()

            #process ranking points
            ranking_points = convertRankingPoints(ranking_points)

            output = {
                'hero_art': hero_match,
                'ranking_points': ranking_points,
                'timestamp': datetime.datetime.utcnow()
            }
            return output
        else:
            latest_HA = hero_match

    cap.release()
    cv2.destroyAllWindows()


def processDir(dirPath):
    with open(os.getcwd() + "/config/box_config.json") as f:
        boxes = json.load(f)
    
    data = []
    for image in tqdm.tqdm(os.listdir(dirPath)):
        # print(dirPath+'/'+image)
        im = cv2.imread(dirPath+'/'+image,0)

        #use different bounding boxes depending on image ratio
        ratio = str(im.shape[1] / im.shape[0])[:3]
        herobox = boxes[ratio]['hero_art']
        pointsbox = boxes[ratio]['ranking_points']
        #right player
        hero_art = processIm(im, herobox)
        ranking_points = processIm(im, pointsbox)
        data.append({
            'hero_art': hero_art,
            'ranking_points': ranking_points,
            'timestamp': datetime.datetime.utcnow()

        })

    return data

def processReplayDir(dirPath):
    #load templates
    with open(os.getcwd() + "/config/hero_arts.json") as f:
        hero_arts = json.load(f)
    winIm = cv2.imread(os.getcwd() + '/dags/deps/win.png',0)
    loseIm = cv2.imread(os.getcwd() + '/dags/deps/lose.png', 0)

    data = []
    for image in tqdm.tqdm(os.listdir(dirPath)):
        # print(dirPath+'/'+image)
        im = cv2.imread(dirPath+'/'+image,0)

        results = processReplayIm(im=im,
                                  winIm=winIm,
                                  loseIm=loseIm,
                                  hero_arts_true=hero_arts)

        data = data + results

    return data

def roundDown(x):
    return int(math.floor(x/100.0) * 100)

def convertRankingPoints(ranking_points):
    #split on p
    ranking_points = ranking_points.split('p')[0]

    #remove all non-numerical characters
    ranking_points = re.sub('[^0-9]','',ranking_points) 
    
    #if empty string then set to 9999 aka champion
    if ranking_points == '' or len(ranking_points) < 4: 
        ranking_points = 0
    else:
        ranking_points = int(ranking_points)

    if ranking_points > 9999:
        ranking_points = 0
    
    return ranking_points


if __name__ == '__main__':
    with open("config/hero_arts.json") as f:
        hero_arts = json.load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="directory for image files")    
    args = parser.parse_args()


    data = processDir(args.dir)
    
    df = pd.DataFrame(data)
    #df.to_csv("{}/ranking-data-raw.txt".format(args.dir), index=False)

    start_size = df.shape[0]
    df.hero_art = df.hero_art.map(cleanString)
    #clean up hero arts with fuzzywuzzy
#    df.hero_art = df[df.hero_art != ' ']

    old_ha = df.hero_art.copy()
    df.hero_art = df.hero_art.map(lambda x: process.extractOne(x, hero_arts.keys())[0])
    
    #debugging for 
    old_ha = pd.concat([old_ha, df.hero_art], axis =1)
    print(old_ha)

    df.ranking_points = df.ranking_points.map(cleanString)
    df.ranking_points = df.ranking_points.map(lambda x: re.sub('[^0-9]','',x))

    #clean up ranking points need to start tracking people below champion somehow
    df.ranking_points = df.ranking_points.map(lambda x: int(x) if x!= '' else 9999)
    df = df[df.ranking_points > 1000]

    end_size = df.shape[0]
    print("Lost {} rows of data out of {} rows".format(start_size-end_size, start_size))
    df.head()

    df.to_csv("{}/ranking-data.txt".format(args.dir), index=False)

    #crosstab for visualization
    df.ranking_points = df.ranking_points.map(roundDown)
    tab = pd.crosstab(df.ranking_points, df.hero_art).transpose()
    tab.to_csv('{}/ranking-crosstab.csv'.format(args.dir))

    #hero counts
    df['hero'] = df.hero_art.map(lambda x: hero_arts[x])
    temp = df.groupby(['hero']).count().add_suffix('_Count').reset_index()
    temp.to_csv('{}/hero_count.csv'.format(args.dir), index=False)


    #hero art counts
    #df.hero = df.hero_art.apply(lambda x: hero_arts_2[x])
    temp = df.groupby(['hero_art']).count().add_suffix('_Count').reset_index()
    temp.to_csv('{}/hero_art_count.csv'.format(args.dir), index=False)
