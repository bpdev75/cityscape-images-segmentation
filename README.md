# cityscape-images-segmentation
AI-based image segmentation for autonomous vehicles using the Cityscapes dataset. This project includes a FastAPI backend for processing images and generating segmentation masks, and a Streamlit frontend for easy interaction and visualization.

## Features
- Image segmentation for urban scenes captured by autonomous cameras.
- Utilizes the **Cityscapes** dataset for model training.
- FastAPI backend for processing images and generating segmentation masks.
- Streamlit frontend for visualizing original images and segmentation results.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/cityscape-images-segmentation.git
    cd cityscape-images-segmentation
    ```

2. **Set up a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run the FastAPI backend:
    ```bash
    uvicorn api.api:app --reload
    ```
    The API will be available at [http://localhost:8000](http://localhost:8000).

### Run the Streamlit frontend:
    ```bash
    streamlit run frontend/app.py
    ```
    The frontend will be available at [http://localhost:8501](http://localhost:8501).

## API Example

**POST** to `/segmentation/`:

**Request body**:
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAA..."
}
