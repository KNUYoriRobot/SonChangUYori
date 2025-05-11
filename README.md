# CookingBot

## Meta2Qeust 통신성공인 이미지파일가져오기

```
docker pull iuiwhgi/meta2quest_noetic_env:latest
```

```
docker run -it --rm \
  --net=host \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  iuiwhgi/meta2quest_noetic_env:latest
```
![image](https://github.com/user-attachments/assets/83e4d38d-93a9-4b37-a6cb-c141e8251b6a)


🎥 [Mujoco 데모 영상 보기](https://youtu.be/vcx9XtKrGoE?si=CAnEslmHp8IWnlmw)



## 환경세팅 성과

dual_robot_control.py 실행하면 아래처럼 뜸. 왼쪽 카메라 시점도 같이 뜬다. (OpenCv의 한계로 캠 화면 꺼도 다시 뜬다)
![Screenshot from 2025-05-05 17-20-40](https://github.com/user-attachments/assets/913a4958-0eb0-4f68-b391-84cd8cf90755)

aloha/load.py 실행시 (기존 알로하 2 수정해서 오픈메니퓰레이터 버전으로 만들기완료)
![Screenshot from 2025-05-05 17-48-52](https://github.com/user-attachments/assets/40c36ed0-ee19-49e1-b83c-8a63d0653785)




















---

# 여기서 밑에는 수정하던  주간고사 이전에 수정하던 IsaacSim 내용들

[🔗 현재 수정중인 문제](https://github.com/iui-whgi/CookingBot/blob/main/URDF/Problem.md)

0405 
.xacro 파일에서 urdf추출하기 성공

![image](https://github.com/user-attachments/assets/7fbbbb7a-5edb-459a-bed4-3cd5b304c0a6)

![Screenshot from 2025-04-05 01-31-55](https://github.com/user-attachments/assets/992026a6-3fb8-4b4b-bb2c-861327da45ee)

![Screenshot from 2025-04-05 01-31-36](https://github.com/user-attachments/assets/abe7eaa8-8c96-4f4d-8e32-c5a8e5f4feee)

![Screenshot from 2025-04-05 01-38-40](https://github.com/user-attachments/assets/9a03bef4-fee0-4242-bdff-a45a8a9ce2c7)




![Screenshot from 2025-04-05 11-58-58](https://github.com/user-attachments/assets/a5098753-da61-41dc-a8a5-8c6ac0b2e657)

[Screencast from 04-05-2025 12:14:06 PM.webm](https://github.com/user-attachments/assets/2109adbe-9ddc-4c07-a56c-a2e60628716b)

---

# Mujoco 버전 3.3.2 씀


![image](https://github.com/user-attachments/assets/d2102733-f83f-45b5-a0cd-27157a977541)
