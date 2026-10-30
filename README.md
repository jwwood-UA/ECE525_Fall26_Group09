# ECE525 Course Project - Group09

This project is an interactive teaching tool explaining, through several visualizations and hands-on explorations, 
how token attention values are formed throughout the transformer and how the model uses them to generate the output.

## How to Run the App
### 1. Set up the environment

Clone the repository and enter the main directory

```bash
git clone https://github.com/jwwood-UA/ECE525_Fall26_Group09.git
cd ECE525_Fall26_Group09
```

We use python 3.13, but other versions of python 3 are likely to work as well. 
Create a local python environment using venv or conda, or use a global python installation.

```bash
# conda
conda create -n ECE525 python=3.13 pip -y
conda activate ECE525
```

```bash
# venv bash
python -m venv .venv
source .venv/bin/activate

# venv powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install web server and backend dependencies

```bash
python -m pip install -r UI/requirements.txt
python -m pip install -r backend/requirements.txt
```

### 2. Running the Server

```bash
python UI/src/app.py
```

or

```bash
uvicorn --app-dir UI/src app:app
```

This method allows you to change certain running parameters like the port using uvicorn's CLI arguments

### 3. Using the App

Open a web browser to ```localhost:8000```

## Implementation Progress
| Milestone | Status |
|---|---|
| FastAPI server (serves UI) | ![Functional](https://img.shields.io/badge/Functional-1f6feb?style=flat-square) |
| GPT-2 logit lens backend | ![Functional](https://img.shields.io/badge/Functional-1f6feb?style=flat-square) |
| UI ↔ backend integration | ![Functional](https://img.shields.io/badge/Functional-1f6feb?style=flat-square) |
| Logit lens grid UI | ![Functional](https://img.shields.io/badge/Functional-1f6feb?style=flat-square) |
| Client round-trip latency measurement | ![Functional](https://img.shields.io/badge/Functional-1f6feb?style=flat-square) |
| Server-side timing breakdown (compute vs. network) | ![Functional](https://img.shields.io/badge/Functional-1f6feb?style=flat-square) |
| Load testing / concurrency limits | ![Not Started](https://img.shields.io/badge/Not%20Started-8b949e?style=flat-square) |
| Activation patching UI | ![Not Started](https://img.shields.io/badge/Not%20Started-8b949e?style=flat-square) |
| Activation patching Backend | ![Not Started](https://img.shields.io/badge/Not%20Started-8b949e?style=flat-square) |
| Guided tutorial content | ![Not Started](https://img.shields.io/badge/Not%20Started-8b949e?style=flat-square) |

**Legend:**  
![Finalized](https://img.shields.io/badge/Finalized-2ea44f?style=flat-square) complete and polished  
![Functional](https://img.shields.io/badge/Functional-1f6feb?style=flat-square) works end-to-end, may still change  
![Started](https://img.shields.io/badge/Started-e3b341?style=flat-square) in progress  
![Not Started](https://img.shields.io/badge/Not%20Started-8b949e?style=flat-square) planned
