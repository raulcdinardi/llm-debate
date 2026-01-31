import asyncio
import atexit
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Proxy configuration - auto-launch mitmweb (enabled by default)
PROXY_PORT = 8080
WEB_PORT = 8081
USE_PROXY = "--no-proxy" not in sys.argv

if "--no-proxy" in sys.argv:
    sys.argv.remove("--no-proxy")

if USE_PROXY:
    PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"

    print(f"\n[mitmproxy] Starting mitmweb on port {PROXY_PORT}...")
    mitmweb_path = Path(sys.executable).parent / "mitmweb"
    assert mitmweb_path.exists(), f"mitmweb not found at {mitmweb_path}. Install with: pip install mitmproxy"
    mitm_proc = subprocess.Popen(
        [str(mitmweb_path), "--listen-port", str(PROXY_PORT), "--web-port", str(WEB_PORT), "--set", "ssl_insecure=true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(mitm_proc.terminate)
    time.sleep(1)

    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL

    # Use mitmproxy's CA certificate
    mitmproxy_ca = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
    if mitmproxy_ca.exists():
        os.environ["SSL_CERT_FILE"] = str(mitmproxy_ca)
        os.environ["REQUESTS_CA_BUNDLE"] = str(mitmproxy_ca)

    print(f"[mitmproxy] Web UI: http://127.0.0.1:{WEB_PORT}\n")
else:
    print("\n[mitmproxy] Disabled via --no-proxy flag.\n")

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    builder = Gsm8kDatasetBuilder(
        batch_size=128,
        group_size=16,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "log_path": "logs/rl_basic",
            "dataset_builder": builder,
            "learning_rate": 4e-5,
            "max_tokens": 256,
            "eval_every": 0,
        }
    )


def main(config: train.Config):
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
