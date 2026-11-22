"""Write experiment events to a local Trackio run in an isolated environment."""
import contextlib
import json
import sys
from importlib import import_module


def main() -> None:
    with contextlib.redirect_stdout(sys.stderr):
        trackio = import_module('trackio')

        config = json.loads(sys.stdin.readline())
        run = trackio.init(
            project=config['project'],
            name=config['name'],
            config=config['config'],
            space_id=None,
            server_url=None,
            dataset_id=None,
            bucket_id=None,
            embed=False,
            auto_log_gpu=False,
            auto_log_cpu=False,
        )
    print('ready', flush=True)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                event = json.loads(line)
                if event['type'] == 'finish':
                    break
                run.log(event['metrics'], step=event.get('step'))
        finally:
            run.finish()


if __name__ == '__main__':
    main()
