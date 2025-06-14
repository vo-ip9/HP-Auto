"""
made by vo-ip9
discord: @vondeehair
tutorial video can be found on yt @Vondee Hair
DM me any bugs or suggestions
"""

import os
import cv2
import time
import ctypes
import warnings
import pyautogui
import pytesseract
import numpy as np
import customtkinter as ctk

from ctypes import wintypes
from rich.console import Console

console = Console()
mouse_pressed = False
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
warnings.filterwarnings("ignore")
failure_counters = {"1": 0, "2": 0, "3": 0, "4": 0}


def get_square_centers_bilinear(corners, grid_size=5):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners

    centers = np.zeros((grid_size, grid_size, 2))
    for i in range(grid_size):
        for j in range(grid_size):
            u = (j + 0.5) / grid_size
            v = (i + 0.5) / grid_size

            # bilinear interpolation (best algorithm for this)
            center_x = (1-u)*(1-v)*x1 + u*(1-v)*x2 + (1-u)*v*x3 + u*v*x4
            center_y = (1-u)*(1-v)*y1 + u*(1-v)*y2 + (1-u)*v*y3 + u*v*y4
            centers[i, j] = [center_x, center_y]

    return centers


def extract_individual_squares(img, corners, grid_size=5):
    centers = get_square_centers_bilinear(corners, grid_size)
    squares = []
    
    # square dimensions from the original grid
    grid_width = max(corners[1][0], corners[3][0]) - min(corners[0][0], corners[2][0])
    grid_height = max(corners[2][1], corners[3][1]) - min(corners[0][1], corners[1][1])
    
    # individual square size
    est_square_width = grid_width / grid_size
    est_square_height = grid_height / grid_size
    
    # use smaller dimension to avoid overlap
    crop_size = int(min(est_square_width, est_square_height))
    
    for i in range(grid_size):
        for j in range(grid_size):
            center_x = int(centers[i, j, 0])
            center_y = int(centers[i, j, 1])
            
            # get crop boundaries
            half_size = crop_size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(img.shape[1], center_x + half_size)
            y2 = min(img.shape[0], center_y + half_size)
            
            # get square image
            square_img = img[y1:y2, x1:x2]
            if square_img.size > 0:
                #square_img = cv2.resize(square_img, (100, 100))
                squares.append((square_img, i, j))
    
    return squares


def process_screenshot_for_ocr(screenshot_region, relative_corners):
    screenshot = pyautogui.screenshot(region=screenshot_region)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    squares = extract_individual_squares(img, relative_corners)
    return squares


def rgb_to_hsv_sum(rgb_list):
    if len(rgb_list) != 3:
        return None

    r, g, b = [x / 255.0 for x in rgb_list]
    cmax, cmin = max(r, g, b), min(r, g, b)
    d = cmax - cmin

    # hue
    h = 0
    if d == 0:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / d) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / d) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / d) + 240) % 360

    # saturation
    s = 0 if cmax == 0 else d * 100 / cmax

    # value
    v = cmax * 100            
    return h + s + v


def directinput_move_to(target_x, target_y, steps=10):
    user32 = ctypes.windll.user32
    for step in range(steps):
        
        # get current position
        current_pos = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(current_pos))
        
        # get remaining distance
        dx = target_x - current_pos.x
        dy = target_y - current_pos.y
        
        # make final jump if cursor is already close
        if abs(dx) <= 2 and abs(dy) <= 2:
            if dx != 0 or dy != 0:
                user32.mouse_event(1, int(dx), int(dy), 0, 0)
            break
        
        # calculate step of the remaining distance
        step_dx = int(dx / (steps - step))
        step_dy = int(dy / (steps - step))

        if step_dx != 0 or step_dy != 0:
            user32.mouse_event(1, step_dx, step_dy, 0, 0)  # MOUSEEVENTF_MOVE
        time.sleep(0.01)


def directinput_click():
    user32 = ctypes.windll.user32
    user32.mouse_event(2, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    time.sleep(0.02)
    user32.mouse_event(4, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


def attempt(numbers, screen_centers):
    # this should be handled during puzzle functions but just in case
    if not all(isinstance(num, (int, float)) for num in numbers.flatten()):
        return

    flat_numbers = numbers.flatten()
    flat_centers = screen_centers.reshape(-1, 2)
   
    numbered_positions = list(enumerate(flat_numbers))
    sorted_positions = sorted(numbered_positions, key=lambda x: x[1])

    for og_idx, number in sorted_positions:
        if number == 0:
            continue
        
        click_coords = flat_centers[og_idx]
        target_x = int(click_coords[0])
        target_y = int(click_coords[1])

        # double click it just to be safe
        directinput_move_to(target_x, target_y)
        directinput_click()
        directinput_click()


def handle_ocr_failure(current_puzzle, next_puzzle_func):
    global failure_counters

    failure_counters[current_puzzle] += 1
    console.print(f"\nFailed reading a tile. Attempt {failure_counters[current_puzzle]}/3", style="bold yellow")

    if failure_counters[current_puzzle] >= 3:
        console.print(f"Too many failures for puzzle {current_puzzle}, moving to next puzzle", style="bold red")
        failure_counters[current_puzzle] = 0
        directinput_move_to(int(middle_square_coords[0]), int(middle_square_coords[1]))
        directinput_click()
        directinput_click()
        directinput_move_to(int(x2)-20, int(y2)-20)
        directinput_click()
        directinput_click()
        directinput_move_to(int(x2), int(y2))
        time.sleep(2)
        os.system("cls")
        next_puzzle_func()
    else:
        console.print("Trying again...", style="bold red")
        directinput_move_to(int(middle_square_coords[0]), int(middle_square_coords[1]))
        directinput_click()
        directinput_click()
        directinput_move_to(int(x2)-20, int(y2)-20)
        directinput_click()
        directinput_click()
        directinput_move_to(int(x2), int(y2))
        time.sleep(2)
        os.system("cls")
        if current_puzzle == "1":
            puzzle_one()
        elif current_puzzle == "2":
            puzzle_two()
        elif current_puzzle == "3":
            puzzle_three()
        elif current_puzzle == "4":
            puzzle_four()


def puzzle_one():
    console.rule("Puzzle One\n", style="bold cyan")
    squares = process_screenshot_for_ocr(screenshot_region, relative_corners)
    console.log("starting OCR engine...")

    reader = easyocr.Reader(["en"], verbose=False)
    numbers = np.zeros((5, 5), dtype=object)
    counter = 1
    
    for img, row, col in squares:
        scale = 3
        h, w = img.shape
        resized = cv2.resize(img, (w*scale,h*scale))
        result = reader.readtext(resized, min_size=5, text_threshold=0.5,
                                low_text=0.3, link_threshold=0.3, allowlist="1234567890\t\n ")
            
        _, text, confidence = result[0] if result else (None, None, 0)
        console.print(f"\r({counter}/25) reading tiles: {text}   ", end="\r")
            
        if text and float(confidence) > 0.5:
            text = "".join([c for c in text if c in "1234567890"])
            numbers[row, col] = int(text)
            counter += 1
        else:
            handle_ocr_failure("1", puzzle_two)
            return

    console.print("\nPressing numbers in order...", style="bold cyan")
    screen_centers = get_square_centers_bilinear(corners)
    attempt(numbers, screen_centers)
    failure_counters["1"] = 0


def puzzle_two():
    console.rule("Puzzle Two\n", style="bold cyan")
    squares = process_screenshot_for_ocr(screenshot_region, relative_corners)
    console.log("starting OCR engine...")
    
    custom_config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=!@#$%^&*()"
    sym_to_num = str.maketrans("!@#$%^&*()", "1234567890")
    numbers = np.zeros((5, 5), dtype=object)
    counter = 1
    
    for img, row, col in squares:
        resized = cv2.resize(img, (100,100))
        text = pytesseract.image_to_string(resized, lang="fivehundredsymbols", config=custom_config)
        
        if not text:
            handle_ocr_failure("2", puzzle_three)
            return

        text = "".join([c for c in text if c in "!@#$%^&*()"])
        translated = text.translate(sym_to_num)
        console.print(f"\r({counter}/25) reading symbols: {text}   ", end="\r")

        if translated:
            numbers[row, col] = int(translated)
            counter += 1

    console.print("\nPressing symbols in order...", style="bold cyan")
    screen_centers = get_square_centers_bilinear(corners)
    attempt(numbers, screen_centers)
    failure_counters["2"] = 0


def puzzle_three():
    console.rule("Puzzle Three\n", style="bold cyan")
    squares = process_screenshot_for_ocr(screenshot_region, relative_corners)
    console.log("starting OCR engine...")

    reader = easyocr.Reader(["en"], verbose=False)
    numbers = np.zeros((5, 5), dtype=object)
    counter = 1
    
    for img, row, col in squares:
        scale = 3
        h, w = img.shape
        resized = cv2.resize(img, (w*scale,h*scale))
        result = reader.readtext(resized, min_size=5, text_threshold=0.5,
                                low_text=0.3, link_threshold=0.3, allowlist="1234567890\t\n ")
            
        _, text, confidence = result[0] if result else (None, None, 0)
        console.print(f"\r({counter}/25) reading tiles: {text}   ", end="\r")
            
        if text and float(confidence) > 0.5:
            text = "".join([c for c in text if c in "1234567890"])
            numbers[row, col] = str(bin(int(text))).count("1")
            counter += 1
        else:
            handle_ocr_failure("3", puzzle_four)
            return

    console.print("\nPressing numbers in order...", style="bold cyan")
    screen_centers = get_square_centers_bilinear(corners)
    attempt(numbers, screen_centers)
    failure_counters["3"] = 0


def puzzle_four():
    console.rule("Puzzle Four\n", style="bold cyan")
    console.log("starting OCR engine...")
    reader = easyocr.Reader(["en"], verbose=False)
    numbers = np.zeros((5,5), dtype=object)
    square_length = relative_corners[1][0] / 5
    square_height = relative_corners[2][1] / 5
    crop_padding = 15
    counter = 1
    square_centers = get_square_centers_bilinear(corners)
   
    for row in range(5):
        for col in range(5):
            mid_x, mid_y = square_centers[row, col]
            cursor_x = mid_x + square_length / 2 - crop_padding
            cursor_y = mid_y + square_height / 2 - crop_padding
            directinput_move_to(int(cursor_x), int(cursor_y))
            region = (int(mid_x-square_length/2), int(mid_y-square_height/2), int(square_length), int(square_height))
            screenshot = pyautogui.screenshot(region=region)
           
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
           
            scale = 3
            h, w = img.shape
            resized = cv2.resize(img, (w*scale, h*scale))
            result = reader.readtext(resized, allowlist="1234567890\n\t\r ", paragraph=False,
                                     detail=1, text_threshold=0.4, low_text=0.2, link_threshold=0.3,
                                     width_ths=0.5, height_ths=0.5, decoder="greedy", beamWidth=3, batch_size=1)
           
            all_text = []
            total_confidence = 0.0
            
            if result:
                for detection in result:
                    if len(detection) == 3:
                        _, text, confidence = detection
                    elif len(detection) == 2:
                        _, text = detection
                        confidence = 1.0
                    else:
                        continue
                    
                    clean_text = "".join(c for c in text if c.isdigit() or c.isspace())
                    if clean_text.strip():
                        all_text.append(clean_text.strip())
                        total_confidence += float(confidence)
                
                combined_text = " ".join(all_text)
            else:
                combined_text = ""
           
            if not combined_text:
                handle_ocr_failure("4", puzzle_one)
                return
                
            rgb_values = []
            for word in combined_text.split():
                if word.isdigit():
                    rgb_values.append(int(word))
            
            if len(rgb_values) != 3:
                digits_only = "".join(c for c in combined_text if c.isdigit())
                if len(digits_only) == 3:
                    rgb_values = [int(d) for d in digits_only]
                
            if len(rgb_values) == 3 and all(0 <= x <= 255 for x in rgb_values):
                console.print(f"\r({counter}/25) reading symbols: {rgb_values}    ", end="\r")
                numbers[row, col] = rgb_to_hsv_sum(rgb_values)
            else:
                console.print(f"\nFailed to parse RGB values from: \"{combined_text}\"", style="bold red")
                handle_ocr_failure("4", puzzle_one)
                return
            counter += 1

    console.print("\nPressing numbers in order...", style="bold cyan")
    screen_centers = get_square_centers_bilinear(corners)
    attempt(numbers, screen_centers)    
    failure_counters["4"] = 0
   
    time.sleep(9)
    directinput_move_to(int(start_button_coords[0]), int(start_button_coords[1]))
    directinput_click()
    directinput_click()
    time.sleep(2)


def run_puzzle(puzzle_type):
    puzzle_functions = {
        "1": puzzle_one,
        "2": puzzle_two, 
        "3": puzzle_three,
        "4": puzzle_four
    }
    
    if puzzle_type in puzzle_functions:
        puzzle_functions[puzzle_type]()
        directinput_move_to(x2, y2)
        time.sleep(2)
        os.system("cls")


def start_gui():
    coords = None
    titlebar_height = None
   
    def change_alpha(_):
        root.attributes("-alpha", alpha_slider.get()/10)
        slider_label.configure(text=f"{(round(alpha_slider.get()*10))}%")
   
    def get_info():
        nonlocal coords
        nonlocal titlebar_height
       
        root.update_idletasks()
        x1, y1 = root.winfo_rootx(), root.winfo_rooty()
        x2 = x1 + root.winfo_width()
        y2 = y1 + root.winfo_height()
       
        root.update_idletasks()
        titlebar_height = root.winfo_rooty() - root.winfo_y()
        coords = (x1, y1, x2, y2)
        root.quit()
        root.destroy()
   
    root = ctk.CTk()
    root.title("HP Setup")
    root.geometry("300x250")
    root.minsize(300, 250)
    root.wm_attributes("-topmost", True)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
   
    instruction_title = ctk.CTkLabel(root, justify="center", text="Instructions", font=("Verdana Bold", 16))
    instruction_label = ctk.CTkLabel(root, justify="center", wraplength=250, font=("Verdana", 12),
                                text="Position and resize this window so it is flush with the "
                                     "puzzle grid.\nAdjust window transparency as needed.\nPress done when ready to screenshot "
                                     "or exit to close the window.")
   
    alpha_frame = ctk.CTkFrame(root)
    alpha_slider = ctk.CTkSlider(alpha_frame, from_=1, to=10, width=200, command=change_alpha)
    slider_title = ctk.CTkLabel(alpha_frame, text="Adjust Transparency", font=("Verdana", 14))
    slider_label = ctk.CTkLabel(alpha_frame, text="100%", font=("Verdana", 12))
   
    button_frame = ctk.CTkFrame(root)
    start_button = ctk.CTkButton(button_frame, text="Done", command=get_info, width=20)
    exit_button = ctk.CTkButton(button_frame, text="Exit", command=root.destroy, width=20)
   
    alpha_slider.set(10)
    instruction_title.pack(pady=5)
    instruction_label.pack()
    alpha_frame.pack(pady=20)
    slider_title.pack()
    alpha_slider.pack(side="left")
    slider_label.pack(padx=5, side="right")
    button_frame.pack()
    start_button.pack(side="left", padx=5, pady=5)
    exit_button.pack(side="right", padx=5, pady=5)

    root.mainloop()
    return coords, titlebar_height


def main():
    os.system("cls")
    while True:
        try:
            for puzzle_type in pattern:
                run_puzzle(puzzle_type)
        except KeyboardInterrupt:
            console.log("exiting...")
            exit()


if __name__ == "__main__":
    os.system("cls")
    pattern = console.input("puzzle pattern (1-2-3-4, 2-3-4, etc): ").split("-")
    console.log("importing easyocr...")
    import easyocr

    console.log("opening GUI...")
    coords, titlebar_height = start_gui()

    if coords and titlebar_height:
        x1, y1, x2, y2 = coords
        y1 -= titlebar_height
        screenshot_region = (x1, y1, x2-x1, y2-y1)
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        relative_corners = [(0,0), (x2-x1,0), (0,y2-y1), (x2-x1,y2-y1)]
        middle_square_coords = (x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.5)
        start_button_coords = (x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.85)
               
        try:
            directinput_move_to(int(x2), int(y2))
            screenshot = pyautogui.screenshot(region=screenshot_region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            console.print(f"error taking screenshot: {e}", style="bold red")
    else:
        console.print("window was closed without getting coordinates. exiting.", style="bold red")
        
    main()