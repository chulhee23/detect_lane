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







## 3. Filtering with Hough transform, Canny edge, Hue Saturation Light (HSL)







## 4. Lane finding with histogram and sliding window









## 5. Finding lane position, and curvature











# Reference

- https://blog.naver.com/hirit808/221486800161
- https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
- https://darkpgmr.tistory.com/31
- 

# Resource

- https://blog.naver.com/hirit808/221486800161