# pylint: disable=subprocess-run-check
import subprocess

from loguru import logger

N_TRIES = 10


class Connection:
    @classmethod
    def warn_if_error(cls, code: int) -> None:
        if code != 0:
            logger.warning(f'Error code: {code}')

    @classmethod
    def send_to_server(
        cls,
        file: str,
        rem_host: str,
        rem_workspace: str,
        recursive: bool = False,
    ) -> None:
        logger.info(f'Sending {file} to {rem_host}:{rem_workspace}/')
        recursive_flag = '-r' if recursive else ''
        Connection.warn_if_error(
            subprocess.run(['scp', recursive_flag, file, f'{rem_host}:{rem_workspace}/']).returncode)

    @classmethod
    def send_content_to_server(cls, file: str, rem_host: str, rem_workspace: str) -> None:
        Connection.warn_if_error(
            subprocess.run(['scp', '-r', f'{file}/*', f'{rem_host}:~/{rem_workspace}/']).returncode)

    @classmethod
    def exec_on_rem_workspace(cls, rem_host: str, rem_workspace: str, cmds: list[str]) -> None:
        cmds = [f'cd {rem_workspace}'] + cmds
        Connection.warn_if_error(subprocess.run(['ssh', rem_host, '; '.join(cmds)]).returncode)
