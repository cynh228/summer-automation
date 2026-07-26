import cv2
import datetime  # 新增

# 1. 打开摄像头（0 代表默认摄像头，如果有多个摄像头就改数字）
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法访问摄像头，请检查驱动或权限")
    exit()

print("按 空格键(Space) 拍照，按 ESC 键退出")

while True:
    # 2. 读取一帧画面（ret 表示是否成功，frame 就是那一帧的图像矩阵）
    ret, frame = cap.read()
    
    if not ret:
        print("获取画面失败")
        break

    # 3. 显示当前画面（窗口名叫做 "Camera"）
    cv2.imshow("Camera", frame)

    # 4. 等待键盘输入（1毫秒检测一次）
    key = cv2.waitKey(1) & 0xFF
    
    # 5. 判断按键
    if key == 27:          # 27 是 ESC 键的 ASCII 码
        print("退出程序")
        break
    elif key == 32:
        # 使用时间戳命名
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{ts}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ 照片已保存为 {filename}")

# 7. 释放摄像头资源（必须写，否则下次调用会出问题）
cap.release()
cv2.destroyAllWindows()