## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the repository

```bash
git clone https://github.com/chatty831/your-repo-name.git
cd your-repo-name
```

### 2. Download the `models` folder

Download the `models` folder from the [Google Drive link](#) (insert your link here), and place it in the root directory of the project.

### 3. Install PyTorch

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and follow the instructions to install the appropriate version of PyTorch for your system.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Set the project root as the working directory

Ensure you're in the project root directory before running any commands:

```bash
cd path/to/project-root
```

### 6. Create a `.env` file

Create a file named `.env` in the root directory and add the following line:

```
FASTAPI_URL="http://localhost:9374"
```

### 7. Start the FastAPI backend

In the terminal, run:

```bash
uvicorn src.app:app --port 9374 --host 0.0.0.0
```

### 8. Start the Streamlit frontend

Open a new terminal window and run:

```bash
streamlit run src/frontend/st_app.py --server.port 8273
```

## 📌 Notes

* Make sure Python 3.10+ is installed.
* Backend will run on `http://localhost:9374`
* Frontend will be available at `http://localhost:8273`