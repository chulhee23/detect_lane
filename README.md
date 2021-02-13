# Traditional Lane Detection



# Project Workflow

- Camera calibration using chessboard
- Perspective transformation to a bird’s eye view
- Filtering with Hough transform, Canny edge, Hue Saturation Light (HSL)
- Lane finding with histogram and sliding window
- Finding lane position, and curvature

## 1. Camera calibration using chessboard

카메라 왜곡은 카메라를 사용할 때 생기는 현상입니다. 흔히 스마트폰의 광각을 사용할 때 쉽게 접하실 수 있는 현상입니다. 이러한 영상 왜곡은 시각적 문제 뿐만 아니라 카메라 왜곡의 현상에는 Radial Distortion(방사 왜곡)과 Tangential Distortion(접선 왜곡)이 존재합니다.

**방사 왜곡**

방사왜곡은 왜곡이 중심에서의 거리에 의해 결정되는 왜곡입니다.

**접선 왜곡**

이미지 촬영 렌즈가 영상의 평면과 완벽하게 평행하지 않기 때문에 접선 왜곡이 발생합니다. 그렇기에, 영상 혹은 이미지의 일부 영역은 보이는 것보다, 예상보다 더 가까워보일 수 있습니다.

이 문서에서는 두 왜곡에 대한 자세한 수식은 생략하였습니다. 
자세한 수식은 https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html 를 참고 부탁드립니다.





## 2. Perspective transformation to a bird’s eye view

차량의 카메라에서 보는 이미지는 전방의 도로 이미지가 원근감을 가지고 있음을 알 수 있습니다.

![image-20210214002545278](readme/camera_shot.png)

원근감이 느껴지는 위의 이미지를 원근감이 없는 이미지로 변환해야 합니다.

이번 프로젝트에서 차선의 위치를 직접 행렬형태로 소스를 잡고 일직선으로 만들 좌표를 만듭니다.
openCV 에서 제공하는 ```getPerspectiveTransform(src, target)``` 함수를 활용하여 변환 행렬을 만들겠습니다.



```python
src = np.array([
                [725, 550],
                [270, 810],
                [920, 550],
                [1250, 810]
], dtype=np.float32)

target = np.array([
                   [495, 0],
                   [495, 810],
                   [1080, 0],
                   [1080, 810]
], dtype=np.float32)


M= cv2.getPerspectiveTransform(src, target)
```



![image-20210214003114922](readme/bird_eye_view.png)



이를 통해 우리는 Bird Eye View 로 변환할 수 있는 행렬을 얻어냈습니다.



## 3. Filtering with Hough transform, Canny edge, Hue Saturation Light (HSL)

가장 중요한 필터링 과정을 진행합니다. 필터링은 이후 차선의 곡률을 구할 때 사용할 슬라이딩 윈도우에 쓸 binray image를 만들기 위해 거치는 과정입니다. 
필터링을 통해 이미지의 노이즈를 줄이고, 차선 인식 정확도를 높이기 위해서 여러가지 방법을 거치게 됩니다.







### Hough Transform









### Canny Edge





### Hue Saturation Light (색상, 채도, 밝기)













## 4. Lane finding with histogram and sliding window









## 5. Finding lane position, and curvature











# Reference

- https://blog.naver.com/hirit808/221486800161
- https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
- https://darkpgmr.tistory.com/31
- 

# Resource

- https://blog.naver.com/hirit808/221486800161