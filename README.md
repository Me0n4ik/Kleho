Here are the detailed installation and launch instructions for both Windows and Ubuntu:

### Windows Instructions

1. **Create Virtual Environment**
```bash
# Open Command Prompt and navigate to your project directory
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

2. **Install Dependencies**
```bash
# With activated virtual environment
pip install -r requirements.txt
```

3. **Install Tesseract OCR**
```bash
# Method 1: Using installer
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
# Run the installer and remember installation path (default: C:\Program Files\Tesseract-OCR)
# Add Tesseract to PATH environment variable:
# Control Panel -> System -> Advanced System Settings -> Environment Variables
# Add C:\Program Files\Tesseract-OCR to Path

# Method 2: Using Chocolatey (if installed)
choco install tesseract
```

4. **Run Program**
```bash
# Make sure virtual environment is activated
python main.py
```

### Ubuntu Instructions

1. **Create Virtual Environment**
```bash
# Open terminal and navigate to your project directory
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

2. **Install Dependencies**
```bash
# With activated virtual environment
pip install -r requirements.txt
```

3. **Install Tesseract OCR**
```bash
# Update package list
sudo apt update

# Install Tesseract and Russian language pack
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-rus
```

4. **Run Program**
```bash
# Make sure virtual environment is activated
python main.py
```

### Troubleshooting

**Windows:**
- If Tesseract is not found, check if the PATH environment variable is set correctly
- Make sure to use the correct Python version (3.7+)
- If pip install fails, try updating pip: `python -m pip install --upgrade pip`

**Ubuntu:**
- If permission denied: use `sudo` for apt commands
- If Python3/pip not found: `sudo apt install python3 python3-pip`
- If virtual environment creation fails: `sudo apt install python3-venv`

### Additional Notes

- Requirements.txt should include:
```
numpy
pandas
pytesseract
pillow
opencv-python
```

- Check Tesseract installation:
```bash
# Windows
tesseract --version

# Ubuntu
tesseract --version
```

- If using VS Code, select the correct Python interpreter (from virtual environment)

- To deactivate virtual environment when done:
```bash
# Both Windows and Ubuntu
deactivate
```

These instructions assume you have Python already installed on your system. If not:
- Windows: Download from python.org
- Ubuntu: Usually pre-installed, if not: `sudo apt install python3` 