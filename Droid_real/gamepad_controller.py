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
        
        # 🔴 核心修复：将转向轴修改为你测出来的 Axis 3
        rx = self.axes.get(3, 0.0)  

        lx = 0.0 if abs(lx) < self.deadzone else lx
        ly = 0.0 if abs(ly) < self.deadzone else ly
        rx = 0.0 if abs(rx) < self.deadzone else rx

        # 前方为正，左侧为正，逆时针转为正
        cmd_x = -ly * 1.0  
        cmd_y = -lx * 0.5  
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
    def get_button_lt(self): 
        # 兼容模式：不管是真实的 Button 8，还是 Axis 2/4 被按下一半以上，都算触发！
        axis_lt = self.axes.get(2, -1.0) > 0.5 or self.axes.get(4, -1.0) > 0.5
        return self.buttons.get(8, False) or axis_lt

    def get_button_rt(self): 
        axis_rt = self.axes.get(5, -1.0) > 0.5
        return self.buttons.get(9, False) or axis_rt

# ==========================================
# 硬件侦测工具 (已增强摇杆轴 Axis 侦测)
# ==========================================
if __name__ == "__main__":
    print("--- 🎮 手柄硬件雷达 ---")
    print("请分别推动左摇杆和右摇杆，观察屏幕上 '摇杆轴' 的变化！")
    print("按 Ctrl+C 退出测试\n")
    
    # 这里把 deadzone 设为 0，为了看清楚最原始的微小数据
    pad = GamepadController(deadzone=0.0) 
    try:
        while True:
            pad.update()
            
            # 获取当前被按下的按钮 ID
            pressed_btns = [k for k, v in pad.buttons.items() if v]
            
            # 获取所有摇杆轴的值 (过滤掉轻微的零点漂移漂移，保留绝对值 > 0.05 的)
            active_axes = {k: round(v, 2) for k, v in pad.axes.items() if abs(v) > 0.05}
            
            # 清除并同行打印
            print(f"\r[雷达] 按键(Btn): {pressed_btns} | 摇杆轴(Axis): {active_axes}                        ", end="")
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] 测试结束，退出雷达。")
        pygame.quit()