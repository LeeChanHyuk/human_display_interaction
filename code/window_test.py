import pyautogui

fore = pyautogui.getActiveWindow()
print(fore.title)
print(fore.size)
print(fore.left)