# Matrix Notation Tool (Notação Matricial)

A Streamlit application that implements the A/B/C model for analyzing symbolic-affective relationships in experiences.

## Features

- Build incidence matrices from experiences, symbols, and emotions
- Compute the symbol-affect field matrix C = A^T B
- Perform SVD analysis to extract dominant modes
- Visualize the "idéia máxima" (maximum idea) - the dominant symbolic-affective pattern
- Create latent space maps for symbols and affects

## Installation

1. Create and activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

After activating the virtual environment, run:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

1. Enter your experiences in the interactive table
2. For each experience, provide:
   - A brief description
   - Comma-separated symbols (e.g., `table, wine, silence`)
   - Comma-separated emotions (e.g., `anger, sadness`)
3. Click "Run analysis" to see:
   - The symbol-affect field matrix C
   - SVD decomposition and energy shares
   - The dominant mode (idéia máxima)
   - Latent space visualizations

## Dependencies

- streamlit >= 1.28.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- plotly >= 5.17.0


