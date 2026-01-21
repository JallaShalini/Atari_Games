#!/usr/bin/env python3
import sys
import importlib

def check_environment():
    print("=" * 50)
    print("DQN ATARI ENVIRONMENT SANITY CHECK")
    print("=" * 50)
    
    required_packages = [
        'torch', 'torchvision', 'gym', 'numpy', 'cv2',
        'pandas', 'matplotlib', 'sklearn', 'fastapi',
        'uvicorn', 'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package:20} OK")
        except ImportError:
            print(f"✗ {package:20} MISSING")
            missing.append(package)
    
    print("=" * 50)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All dependencies installed!")
        return True

if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)
