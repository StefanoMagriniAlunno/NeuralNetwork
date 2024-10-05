import sys

from invoke import task  # type: ignore

list_packages = [
    "jupyter",
    "pandas tqdm tabulate types-tabulate",  # utilities
    "torch torchvision --index-url https://download.pytorch.org/whl/cu124",  # gpu computing
    "matplotlib seaborn plotly",  # visualizzazione
]


@task
def download(c, cache: str):
    """download contents"""

    for pkg in list_packages:
        print(f" - Downloading {pkg}")
        try:
            c.run(
                f"{sys.executable} -m pip download --no-cache-dir --dest {cache} --quiet {pkg} "
            )
            print("✅")
        except Exception as e:
            print("❌")
            print(e)
            raise e


@task
def install(c, cache: str):
    """Install contests"""

    for pkg in list_packages:
        print(f"Installing {pkg}")
        try:
            c.run(
                f"{sys.executable} -m pip install --compile --no-index --find-links={cache} --quiet {pkg} "
            )
            print("✅")
        except Exception as e:
            print("❌")
            print(e)
            raise e
