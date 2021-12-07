# Tools and Wrappers used for Durian's Projects

### Utils
Quality of life functions to avoid a lot of repeat code. 

### Image_analysis
functions related to pre-processing, manipulating and processing images specifically in order to extract TEPPEN game information from them.

### Twitch
API Wrapper for twitch used to download streams/clips to files/urls.

## Workflow
Use twitch + utils to download streams/clips or turn them into datastreams.

Use Image analysis to extract relevant TEPPEN game informations out of datastreams/video files or static image files

# TODO:
1. file renaming to make modules names more straight forwards ie: utils is currently just streamlink utils
2. directory organization to separate api wrappers, various utilities and project specific code (ie: image_analysis has many teppen specific functions, the teppen specific code may be better off in a different package all together?)
3. ADD MORE FUN :) 
