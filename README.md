# Shoppin_Task

***Objective***
- Detect shoppable items in a 60-second video while minimizing computational time by selecting around 20 frames out of 900 for object detection.
- Propose methods to optimize the frame selection process, reducing the total number of frames for object detection from 900 to 20, while maintaining high accuracy in detecting shoppable items.

***Test videos I have used***

test_video1 :- 1minute video from a youtube video by a fashion youtuber

test_video2:- 1minute video from Armani Men fashion walk youtube video

test_video3:- 16second video which is created by combining two clothing transition reels (The Most difficult test video)

(NOTE: All above videos is 30Fps)

***Approaches tried for frame selection***

- Motion intensity based approach

  Codefile:- motion_approach.py
  
  Find the selected frames output from test videos in folders:- motion_selected_frames_testvid1, motion_selected_frames_testvid2 and motion_selected_frames_testvid3
  
- Content detector (from Pyscenedetect)
  
  Codefile:- content_detector.py
  
  Find the selected frames output from test videos in folders:- content_selected_frames_testvid1, content_selected_frames_testvid2 and content_selected_frames_testvid3
  
- Adaptive detector (from Pyscenedetect)

  Codefile:- adaptive_detector.py
  
  Find the selected frames output from test videos in folders:- adaptive_selected_frames_testvid1, adaptive_selected_frames_testvid2 and adaptive_selected_frames_testvid3
  
- Histogram detector (from Pyscenedetect)

  Codefile:- histogram_detector.py
  
  Find the selected frames output from test videos in folders:- histogram_selected_frames_testvid1, histogram_selected_frames_testvid2 and histogram_selected_frames_testvid3

***Best Frame Selection approach***

Since here we do not have any ground truth labels, any quantitative measure is not possible. Only Qualitative analysis can be used to decide the best performing approach. 

I suggest the reader to first look all three test videos and then see the frames selected by each approach from the folders. You will notice that best performing are motion based approach and adaptive detector. Both have almost similar selection of frames but the only limitation in Motion based is that it does not capture the frame that comes first (i.e frame at the start of the video).

Best results were obtained by **Adaptive Detector**

  


