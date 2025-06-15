# HP-Auto
Fully Automatic "Harry's Puzzle" Solver for the Roblox game [Everything Upgrade Tree](https://www.roblox.com/games/122809141833750/Everything-Upgrade-Tree)

# Installation
Note: This script will only work on machines running Windows, as it uses the win32 API.

1. Download TesseractOCR from the [main repository](https://github.com/tesseract-ocr/tesseract)

2. Clone this repository or download the latest release .zip (version 1.0)
```
git clone https://github.com/vo-ip9/HP-Auto
```

3. Move `hp_symbols.traineddata` to your `..\Tesseract-OCR\tessdata` directory

4. Copy the path of `..\Tesseract-OCR\tesseract.exe` and paste into main.py, line 23

5. Install the requirements and run the script (main.py)
```
pip install -r requirements.txt
python main.py
```
