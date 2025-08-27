# Intro
Project bootstrapper for Maya rig/animation tooling.

- Creates directory dependency: config/, logs/, cache/, exports/, temp/, scripts/
- Spawns venv and wires .env
- Generates VSCode settings/launch/tasks for mayapy, unit tests
- Setup git config and .gitignore
- Writes ready-to-use pipeline skeleton:
    scripts/<pkg>/{__init__.py, exceptions.py, logger.py, filesys.py, guards.py}
    config/{studio.toml, user.toml}
    modules/<Pkg>.mod   (Maya module file)

# Usage
python MLsetup.py --name MlRigSys --dest E: --pkg rigkit --devkit ""
