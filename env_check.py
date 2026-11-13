import sys
import platform
import importlib.metadata
from datetime import datetime
from pathlib import Path

# Required packages
packages = {
    "torch": "torch",
    "ultralytics": "ultralytics",
    "streamlit": "streamlit",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python-multipart": "multipart",
    "pandas": "pandas",
    "pillow": "PIL"
}

report = []

def add_line(text):
    report.append(text)
    print(text)


# Report heading
add_line("# VisionPort Environment Report")
add_line(f"\nGenerated: {datetime.now()}")
add_line(f"\nPython Version: {sys.version}")
add_line(f"\nPython Executable: {sys.executable}")
add_line(f"\nOperating System: {platform.platform()}")

# Virtual environment check
add_line("\n## Virtual Environment")

if sys.prefix != sys.base_prefix:
    add_line("- Status: Active")
else:
    add_line("- Status: Not Active")

# Package verification
add_line("\n## Package Verification")

for package, module in packages.items():
    try:
        version = importlib.metadata.version(package)
        add_line(f"- {package}: Installed (Version {version})")
    except importlib.metadata.PackageNotFoundError:
        add_line(f"- {package}: NOT INSTALLED")

# PyTorch verification
add_line("\n## PyTorch Configuration")

try:
    import torch

    add_line(f"- PyTorch Version: {torch.__version__}")
    add_line(f"- CUDA Version: {torch.version.cuda}")
    add_line(f"- CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        add_line(f"- GPU: {torch.cuda.get_device_name(0)}")
    else:
        add_line("- GPU: Not available (CPU mode)")

except Exception as error:
    add_line(f"- PyTorch Error: {error}")

# Save report
report_path = Path(__file__).parent / "env_report.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("\n--------------------------------")
print("Environment check completed!")
print(f"Report saved to: {report_path}")
print("--------------------------------")