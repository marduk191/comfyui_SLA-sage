"""
Installation script for ComfyUI SLA-sage nodes
This script installs the required dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("=" * 60)
    print("Installing ComfyUI SLA-sage requirements...")
    print("=" * 60)

    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if not os.path.exists(requirements_path):
        print("ERROR: requirements.txt not found!")
        return False

    try:
        # Install requirements
        print("\nInstalling dependencies from requirements.txt...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_path
        ])

        print("\n" + "=" * 60)
        print("Installation completed successfully!")
        print("=" * 60)
        print("\nYou can now use SLA nodes in ComfyUI:")
        print("  - SLA Attention")
        print("  - SageSLA Attention (Optimized)")
        print("\nRestart ComfyUI to load the nodes.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Installation failed: {e}")
        print("\nYou can try manual installation:")
        print("  pip install git+https://github.com/marduk191/SLA.git")
        return False

if __name__ == "__main__":
    success = install_requirements()
    sys.exit(0 if success else 1)
