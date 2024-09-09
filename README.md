## Setup project
- Setup black, isort, flake and pre-commit by following the instructions in this [link](https://viblo.asia/p/format-code-python-tu-dong-su-dung-isort-black-flake8-va-pre-commit-3P0lPDEolox).

## Commands to setup
We have implemented our work on Python 3.9

Install needed packages
```bash
cd ABC
pip install -r requirements.txt
```
## Task
Update 09/09
- [ ] Implement Image encoder
- [ ] Implement CL for visual
- [ ] Implement dataset for garment-garment matching
    - [ ] Write dataset to load garment pairs
    - [ ] Add global geometric augmentation (rotate, shear, upide down, translation) (=> positive pair)
    - [ ] Add perturbation (delete pattern, create artifact, change color, blur, flip, color distortion, jisaw) into garment positions (=> hard negative pair)
- [ ] Training Image encoder for garment matching
