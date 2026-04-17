import os
os.environ["SDL_VIDEODRIVER"] = "dummy" # 强制声明使用虚拟视频驱动，防止 Pygame 卡死在寻找显示器上！
import pygame
import time

class GamepadController:
    def __init__(self, deadzone=0.15):
        # 不要用 pygame.init() 全局初始化！
        pygame.display.init()   # 仅初始化事件队列所需的基础模块
        pygame.joystick.init()  # 仅初始化手柄模块
        
        self.joystick = None
        self.deadzone = deadzone

        # 建立状态字典
        self.axes = {}
        self.buttons = {}

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[INFO] 🎮 手柄已连接: {self.joystick.get_name()}")
            for i in range(self.joystick.get_numaxes()): self.axes[i] = 0.0
            for i in range(self.joystick.get_numbuttons()): self.buttons[i] = False
        else:
            print("[WARNING] ⚠️ 未检测到手柄！")

    def update(self):
        """核心：通过事件队列更新所有按键和摇杆状态"""
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.axes[event.axis] = event.value
            elif event.type == pygame.JOYBUTTONDOWN:
                self.buttons[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                self.buttons[event.button] = False

    def get_commands(self):
        """获取并转换速度指令"""
        self.update()
        if self.joystick is None: return 0.0, 0.0, 0.0

        # 严格按照你的硬件图谱映射！
        lx = self.axes.get(0, 0.0)  # 左摇杆 左右 (Axis 0)
        ly = self.axes.get(1, 0.0)  # 左摇杆 上下 (Axis 1)
        rx = self.axes.get(2, 0.0)  # 右摇杆 左右 (Axis 2) 👈 真正控制转向的轴！

        lx = 0.0 if abs(lx) < self.deadzone else lx
        ly = 0.0 if abs(ly) < self.deadzone else ly
        rx = 0.0 if abs(rx) < self.deadzone else rx

        # 前方为正，左侧为正，逆时针转为正
        cmd_x = -ly * 1.0  
        cmd_y = -lx * 0.5  
        # 如果发现左推变成了右转，就把下面这行的负号去掉！
        cmd_yaw = -rx * 1.0 

        return cmd_x, cmd_y, cmd_yaw

    # ==========================================
    # 全按键字典封装 (严格对齐你的硬件)
    # ==========================================
    def get_button_a(self): return self.buttons.get(0, False)
    def get_button_b(self): return self.buttons.get(1, False)
    def get_button_x(self): return self.buttons.get(3, False)
    def get_button_y(self): return self.buttons.get(4, False)
    def get_button_lb(self): return self.buttons.get(6, False)
    def get_button_rb(self): return self.buttons.get(7, False)
    
    # 因为你的LT/RT是模拟按键，所以直接读 button 而不是 axis
    def get_button_lt(self): return self.buttons.get(8, False) 
    def get_button_rt(self): return self.buttons.get(9, False)
    
    def get_button_back(self): return self.buttons.get(10, False)
    def get_button_start(self): return self.buttons.get(11, False)

# ==========================================
# 硬件侦测工具
# ==========================================
if __name__ == "__main__":
    print("--- 🎮 手柄硬件雷达 ---")
    pad = GamepadController(deadzone=0.1)
    try:
        while True:
            pad.update()
            pressed = [k for k, v in pad.buttons.items() if v]
            print(f"\r[雷达] 当前按下的按键 ID: {pressed}                          ", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pygame.quit()