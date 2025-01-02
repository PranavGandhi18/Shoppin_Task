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

Since here we do not have any ground truth frames, any quantitative measure is not possible. Only Qualitative analysis can be used to decide the best performing approach. 

I suggest the reader to first watch all three test videos and then see the frames selected by each approach from the folders. You will notice that best performing are motion based approach and adaptive detector. Both have almost similar selection of frames but the only limitation in Motion based is that it does not capture the frame that comes first (i.e frame at the start of the video).

Best results were obtained by **Adaptive Detector**

***Object detection***

This model from hugging face was used:- https://huggingface.co/yainage90/fashion-object-detection

The labels are ['bag', 'bottom', 'dress', 'hat', 'shoes', 'outer', 'top']

Using Adaptive detector for frame selection and then applying object detection on selected frames. Code can be found in selection_detection.py file. Output of this code can be found in processed_frames_testvid1, processed_frames_testvid2 and processed_frames_testvid3.


***Time difference:- frame selection and detection approach VS. Detection on all frames approach***

**Selection and detection time taken on test video 1**

![Selection and detection time taken on test video 1](https://github.com/PranavGandhi18/Shoppin_Task/blob/main/testvid1.png)

**Detection time taken on all frames of test video 1**

![Detection time taken on all frames of test video 1](https://github.com/PranavGandhi18/Shoppin_Task/blob/main/testvid1_allframes.png)

**Selection and detection time taken on test video 2**

![Selection and detection time taken on test video 2](https://github.com/PranavGandhi18/Shoppin_Task/blob/main/testvid2.png)

**Detection time taken on all frames of test video 2**

![Detection time taken on all frames of test video 2](https://github.com/PranavGandhi18/Shoppin_Task/blob/main/testvid2_allframes.png)

**Selection and detection time taken on test video 3**

![Selection and detection time taken on test video 3](https://github.com/PranavGandhi18/Shoppin_Task/blob/main/testvid3.png)

**Detection time on all frames of test video 3**

![Detection time taken on all frames of test video 2](https://github.com/PranavGandhi18/Shoppin_Task/blob/main/testvid3_allframes.png)


***Suggestions to improve this work in future***

Limitation of the current work:- Look at frames selected for test video 2 in folder processed_frames_testvid2. The clothing products are not at all visible in Frame numbers 2,6 and 19. Such frames that does not have any clothing products should be removed from our selected frames. 

Idea:- Model assisted frame selection

We can use lightweight pretrained model (like:- mobilenet) and finetune few layers at the end on task to score the shoppability of the given frame. In simple words, model will learn to give a probability value which will be closer to 1 if there are shoppable items in the frame. The probability value will be closer to 0 if the frame does not have any shoppable clothing products.

This fine-tuned model must be then used on frames selected by my current method to further filter out frames that does not have any clothing products.

To finetune this model, we would need data of some images with clothing products and without them.

***References***

https://github.com/Breakthrough/PySceneDetect/tree/main

https://www.scenedetect.com/docs/latest/api/detectors.html#scenedetect.detectors.content_detector.ContentDetector


  


