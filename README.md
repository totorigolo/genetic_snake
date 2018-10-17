genetic_snake
=============

# Installation

Create the venv and install the dependencies:
```
python3 -m venv ./venv
source ./venv/bin/activate
pip install colorlog autobahn numpy overrides tensorflow
```



# Usage

Execute the module GA_snake, using `--help` for available options.

Example:
```
python -m ga_snake -r 127.0.0.1 -p 8080 -l info
```
