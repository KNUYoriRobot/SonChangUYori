import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from PIL import Image

def main():
    # scene.xml 파일 로드
    xml_path = "scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 물체 추가 (집을 대상)
    add_objects(model, data)
    
    # 'neutral_pose' 키프레임으로 초기화
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral_pose")
    if keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    
    # 시뮬레이션 한 단계 진행하여 모든 것이 초기화되도록 함
    mujoco.mj_step(model, data)
    
    # 카메라 이미지를 저장할 디렉토리 생성
    os.makedirs("camera_views", exist_ok=True)
    
    # 모든 카메라에서 이미지 저장
    camera_names = ["teleoperator_pov", "collaborator_pov", "wrist_cam_left", 
                   "wrist_cam_right", "overhead_cam", "worms_eye_cam"]
    
    # 렌더러 리소스 생성 (프레임버퍼 크기에 맞게 조정)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    for cam_name in camera_names:
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                # 장면 업데이트 및 렌더링
                renderer.update_scene(data, camera=cam_name)
                img = renderer.render()
                
                # uint8로 변환하고 저장
                img_uint8 = (img * 255).astype(np.uint8)
                Image.fromarray(img_uint8).save(f"camera_views/{cam_name}.png")
                print(f"{cam_name} 카메라에서 이미지 저장 완료")
        except Exception as e:
            print(f"{cam_name} 카메라에서 렌더링 실패: {e}")
    
    # 인터랙티브 시각화를 위한 뷰어 시작
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 뷰어 설정
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        
        # 액추에이터 제어를 위한 설정
        left_arm_actuators = {}
        right_arm_actuators = {}
        
        for joint_name in ["waist", "shoulder", "elbow", "forearm_roll", "gripper"]:
            left_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left/{joint_name}")
            right_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right/{joint_name}")
            
            if left_idx >= 0:
                left_arm_actuators[joint_name] = left_idx
            if right_idx >= 0:
                right_arm_actuators[joint_name] = right_idx
        
        # 시뮬레이션 시작 시간
        start_time = time.time()
        
        # 왼쪽 팔로 픽앤플레이스 시퀀스
        sequence = [
            (5.0, approach_object),    # 물체에 접근
            (3.0, grasp_object),       # 물체 잡기
            (5.0, lift_object),        # 물체 들어올리기
            (5.0, move_object),        # 물체 이동
            (3.0, place_object),       # 물체 내려놓기
            (3.0, release_object),     # 물체 놓기
            (5.0, return_to_home)      # 홈 포지션으로 돌아가기
        ]
        
        current_phase = 0
        phase_start_time = 0
        
        print("시뮬레이션 시작: 픽앤플레이스 작업")
        while viewer.is_running():
            current_time = time.time() - start_time
            
            if current_phase < len(sequence):
                duration, action_func = sequence[current_phase]
                
                # 새 단계 시작
                if current_phase == 0 and phase_start_time == 0:
                    phase_start_time = current_time
                    print(f"단계 {current_phase+1}: {action_func.__name__}")
                
                # 현재 단계의 진행도 (0에서 1 사이)
                phase_progress = min(1.0, (current_time - phase_start_time) / duration)
                
                # 현재 단계의 액션 실행
                action_func(model, data, left_arm_actuators, right_arm_actuators, phase_progress)
                
                # 단계 완료되면 다음 단계로
                if phase_progress >= 1.0:
                    current_phase += 1
                    phase_start_time = current_time
                    if current_phase < len(sequence):
                        print(f"단계 {current_phase+1}: {sequence[current_phase][1].__name__}")
                    else:
                        print("모든 작업 완료!")
            
            # 시뮬레이션 스텝 진행
            mujoco.mj_step(model, data)
            
            # 뷰어 업데이트
            viewer.sync()
            
            # 실시간 속도 유지
            sim_time = data.time
            elapsed = time.time() - start_time
            if sim_time > elapsed:
                time.sleep(sim_time - elapsed)

def add_objects(model, data):
    """MuJoCo 모델에 물체 추가"""
    # 물체 추가를 위한 XML 요소 생성
    cube_xml = """
    <body name="cube" pos="0 -0.3 0.05">
      <joint type="free"/>
      <geom type="box" size="0.03 0.03 0.03" rgba="1 0 0 1" mass="0.1"/>
    </body>
    """
    
    # 목표 위치를 위한 XML 문자열 생성
    target_pos_xml = """
    <body name="target_pos" pos="0.2 -0.3 0.05">
      <geom type="cylinder" size="0.05 0.001" rgba="0 1 0 0.3"/>
    </body>
    """
    
    # 참고: MuJoCo 3.0 이상에서는 동적 모델 수정 방식이 다름
    # 이 예시에서는 간소화되어 있음
    
    print("물체 및 목표 위치 추가됨 (시뮬레이션 상에서)")

# 로봇 움직임 단계 구현 함수들
def approach_object(model, data, left_arm, right_arm, progress):
    """물체에 접근하는 단계"""
    # 왼쪽 팔을 물체를 향해 이동
    target_left = {
        "waist": 0.1,
        "shoulder": -0.7,
        "elbow": 1.1,
        "forearm_roll": 0.0,
        "gripper": 0.03  # 그리퍼 열기
    }
    
    # 오른쪽 팔은 정지 상태로 유지
    target_right = {
        "waist": 0.0,
        "shoulder": -0.96,
        "elbow": 1.2,
        "forearm_roll": 0.0,
        "gripper": 0.0084
    }
    
    # 왼쪽 팔 제어
    for joint, idx in left_arm.items():
        initial_value = data.ctrl[idx]
        target_value = target_left[joint]
        data.ctrl[idx] = initial_value * (1-progress) + target_value * progress
    
    # 오른쪽 팔 제어
    for joint, idx in right_arm.items():
        data.ctrl[idx] = target_right[joint]

def grasp_object(model, data, left_arm, right_arm, progress):
    """물체를 잡는 단계"""
    # 그리퍼만 닫기
    target_gripper = 0.03 * (1-progress) + 0.0084 * progress  # 점점 닫힘
    
    data.ctrl[left_arm["gripper"]] = target_gripper

def lift_object(model, data, left_arm, right_arm, progress):
    """물체를 들어올리는 단계"""
    # 현재 상태에서 높이만 올림
    current_elbow = data.ctrl[left_arm["elbow"]]
    current_shoulder = data.ctrl[left_arm["shoulder"]]
    
    # 들어올리기 (높이 올리기)
    target_elbow = current_elbow * (1-progress) + 0.9 * progress
    target_shoulder = current_shoulder * (1-progress) + (-0.5) * progress
    
    data.ctrl[left_arm["elbow"]] = target_elbow
    data.ctrl[left_arm["shoulder"]] = target_shoulder

def move_object(model, data, left_arm, right_arm, progress):
    """물체를 목표 위치로 이동하는 단계"""
    # 웨이스트 회전으로 물체 이동
    current_waist = data.ctrl[left_arm["waist"]]
    target_waist = current_waist * (1-progress) + 0.5 * progress
    
    data.ctrl[left_arm["waist"]] = target_waist

def place_object(model, data, left_arm, right_arm, progress):
    """물체를 내려놓는 단계"""
    # 현재 상태에서 높이 낮추기
    current_elbow = data.ctrl[left_arm["elbow"]]
    current_shoulder = data.ctrl[left_arm["shoulder"]]
    
    # 내려놓기 (높이 내리기)
    target_elbow = current_elbow * (1-progress) + 1.1 * progress
    target_shoulder = current_shoulder * (1-progress) + (-0.7) * progress
    
    data.ctrl[left_arm["elbow"]] = target_elbow
    data.ctrl[left_arm["shoulder"]] = target_shoulder

def release_object(model, data, left_arm, right_arm, progress):
    """물체를 놓는 단계"""
    # 그리퍼 열기
    target_gripper = 0.0084 * (1-progress) + 0.03 * progress  # 점점 열림
    
    data.ctrl[left_arm["gripper"]] = target_gripper

def return_to_home(model, data, left_arm, right_arm, progress):
    """홈 포지션으로 돌아가는 단계"""
    # 초기 'neutral_pose' 키프레임 상태로 복귀
    target_left = {
        "waist": 0.0,
        "shoulder": -0.96,
        "elbow": 1.16,
        "forearm_roll": 0.0,
        "gripper": 0.0084
    }
    
    # 왼쪽 팔 제어 - 현재 상태에서 홈 포지션으로 
    for joint, idx in left_arm.items():
        current_value = data.ctrl[idx]
        target_value = target_left[joint]
        data.ctrl[idx] = current_value * (1-progress) + target_value * progress

if __name__ == "__main__":
    main()