# import numpy as np
# from scipy.spatial.transform import Rotation as R

# def quat_to_offset(quat, max_angle=(30, 20)):
#     r = R.from_quat(quat)
#     roll, pitch, yaw = r.as_euler('xyz', degrees=True)
#     dx = np.clip(yaw / max_angle[0], -1, 1)
#     dy = np.clip(-pitch / max_angle[1], -1, 1)
#     return dx, dy

# # 模拟几个姿态测试
# test_angles = [
#     (0, 0, 0),       # 正前方
#     (0, 40, 0),      # 低头10°
#     (0, -40, 0),     # 抬头10°
#     (0, 0, 30),      # 向右15°
#     (0, 0, -15),     # 向左15°
# ]

# for euler in test_angles:
#     quat = R.from_euler('xyz', euler, degrees=True).as_quat()
#     dx, dy = quat_to_offset(quat)
#     print(f"quat={quat}°Euler={euler}° → dx={dx:.2f}, dy={dy:.2f}")

import numpy as np
import cv2
import time
from multiprocessing import shared_memory

def update_move_image_from_tv(tv_img_array, move_image_img_array, dx, dy):
    """根据偏移 (dx, dy) 从 tv_img_array 中裁剪局部区域复制到 move_image_img_array"""
    H, W, _ = tv_img_array.shape
    h, w, _ = move_image_img_array.shape

    # 将偏移 [-1, 1] 映射到像素坐标
    x0 = int((W - w) * (dx + 1) / 2)
    y0 = int((H - h) * (dy + 1) / 2)

    # 限制范围，避免越界
    x0 = np.clip(x0, 0, W - w)
    y0 = np.clip(y0, 0, H - h)

    # 裁剪并复制到共享内存中
    sub_image = tv_img_array[y0:y0 + h, x0:x0 + w, :]
    np.copyto(move_image_img_array, sub_image)

def main():
    # --- 1️⃣ 创建共享内存并分配 array ---
    tv_img_shape = (480, 640, 3)  # 模拟全景图像
    move_image_img_shape = (240, 320, 3)  # 移动窗口

    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    move_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(move_image_img_shape) * np.uint8().itemsize)

    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)
    move_image_img_array = np.ndarray(move_image_img_shape, dtype=np.uint8, buffer=move_img_shm.buf)

    # --- 2️⃣ 模拟全景画面内容 ---
    for i in range(tv_img_shape[0]):
        color_value = int(255 * i / tv_img_shape[0])
        tv_img_array[i, :, :] = [color_value, 255 - color_value, 128]

    # --- 3️⃣ 模拟头部移动 ---
    dx, dy = 0.0, 0.0
    t = 0

    try:
        while True:
            # 模拟一个随时间变化的偏移 (摇头 + 点头)
            dx = np.sin(t) * 0.8  # 左右 [-0.8, 0.8]
            dy = np.cos(t / 2) * 0.6  # 上下 [-0.6, 0.6]
            t += 0.05

            # 更新局部图像到共享内存
            update_move_image_from_tv(tv_img_array, move_image_img_array, dx, dy)

            # --- 4️⃣ 显示两个窗口 ---
            display = tv_img_array.copy()
            h, w, _ = move_image_img_array.shape
            H, W, _ = tv_img_array.shape

            # 在主画面上绘制当前裁剪窗口位置
            x0 = int((W - w) * (dx + 1) / 2)
            y0 = int((H - h) * (dy + 1) / 2)
            cv2.rectangle(display, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)

            cv2.imshow("TV Image (Main View)", display)
            cv2.imshow("Move Image (Sub View)", move_image_img_array)
            if cv2.waitKey(30) == 27:  # 按 ESC 退出
                break

    finally:
        # --- 5️⃣ 清理共享内存 ---
        cv2.destroyAllWindows()
        tv_img_shm.close()
        tv_img_shm.unlink()
        move_img_shm.close()
        move_img_shm.unlink()

if __name__ == "__main__":
    main()

