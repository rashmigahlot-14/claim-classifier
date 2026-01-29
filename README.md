
### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```
### 3. Run the Server

```bash
# Option 1: Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Using the Python script
python app.py
```

The API will be available at: `http://localhost:8000`


