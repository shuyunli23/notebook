防止 Windows 息屏的 Python 脚本 keep_awake.pyw (静默运行)
```
import ctypes
import time

# 定义系统常量
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    """
    通知系统保持唤醒状态，防止息屏或睡眠
    """
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )

def allow_sleep():
    """
    恢复系统默认的息屏行为
    """
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

if __name__ == "__main__":
    try:
        print("🟢 防息屏脚本已启动，按 Ctrl+C 可退出")
        while True:
            prevent_sleep()  # 每次调用都“续命”
            time.sleep(60)   # 每分钟执行一次就够了
    except KeyboardInterrupt:
        allow_sleep()
        print("\n🔵 已恢复系统默认息屏策略。")

```
把该脚本放在：
```
C:\Users\<你的用户名>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```
系统登录时会自动启动该脚本。