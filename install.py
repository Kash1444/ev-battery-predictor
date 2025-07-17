#!/usr/bin/env python3
"""
Installation and Setup Script for EV Battery Predictor
TATA Technologies Hackathon Project

Run this script to set up the environment and install dependencies
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("🚗⚡ EV BATTERY DIGITAL TWIN & RANGE PREDICTOR")
    print("=" * 60)
    print("TATA Technologies Hackathon Project")
    print("Automated Installation & Setup")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def get_activation_command():
    """Get the appropriate activation command for the platform"""
    if platform.system() == "Windows":
        return r"venv\Scripts\activate"
    else:
        return "source venv/bin/activate"

def install_requirements():
    """Install required packages"""
    try:
        # Determine the correct pip path
        if platform.system() == "Windows":
            pip_path = os.path.join("venv", "Scripts", "pip")
        else:
            pip_path = os.path.join("venv", "bin", "pip")
        
        print("📋 Installing required packages...")
        
        # Upgrade pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("✅ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False
    except FileNotFoundError:
        print("❌ Virtual environment not found. Please create it first.")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    try:
        if platform.system() == "Windows":
            python_path = os.path.join("venv", "Scripts", "python")
        else:
            python_path = os.path.join("venv", "bin", "python")
        
        print("🔍 Verifying installation...")
        
        # Test imports
        test_script = """
import numpy
import pandas
import sklearn
import matplotlib
import seaborn
print("✅ All core packages imported successfully")
"""
        
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Installation verification passed")
            return True
        else:
            print("❌ Installation verification failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def create_data_directories():
    """Create necessary data directories"""
    directories = ["data", "models", "logs", "outputs"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def print_next_steps():
    """Print instructions for next steps"""
    activation_cmd = get_activation_command()
    
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next steps:")
    print(f"1. Activate virtual environment: {activation_cmd}")
    
    if platform.system() == "Windows":
        print("2. Run the demo: python demo.py")
        print("3. Start the web app: streamlit run app.py")
        print("4. Open Jupyter notebooks: jupyter notebook notebooks/")
    else:
        print("2. Run the demo: python demo.py")
        print("3. Start the web app: streamlit run app.py")
        print("4. Open Jupyter notebooks: jupyter notebook notebooks/")
    
    print("\n📚 Documentation:")
    print("- README.md: Project overview and features")
    print("- notebooks/: Interactive analysis and demos")
    print("- src/: Source code modules")
    
    print("\n💡 Quick Demo:")
    print("  python demo.py")
    
    print("\n🌐 Web Interface:")
    print("  streamlit run app.py")
    print("  Then open: http://localhost:8501")

def main():
    """Main installation function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Installation completed with warnings")
    
    # Create directories
    create_data_directories()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)
