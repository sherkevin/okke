import sys, os
sys.path.insert(0, "/root/autodl-tmp/BRA_Project/MiniGPT-4")
os.chdir("/root/autodl-tmp/BRA_Project/MiniGPT-4")

try:
    from minigpt4.common.config import Config
    print("Config OK")
except Exception as e:
    print(f"Config FAIL: {e}")

try:
    from minigpt4.models.mini_gpt4 import MiniGPT4
    print("MiniGPT4 model class OK")
except Exception as e:
    print(f"MiniGPT4 FAIL: {e}")

try:
    from minigpt4.common.registry import registry
    model_cls = registry.get_model_class("minigpt4")
    print(f"Registry model class: {model_cls}")
except Exception as e:
    print(f"Registry FAIL: {e}")
