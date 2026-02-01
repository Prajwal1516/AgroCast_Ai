# AgroCast Ai 

**AgroCast AI** is an intelligent commodity price forecasting tool designed to empower farmers and traders with accurate, data-driven price predictions. Leveraging deep learning (LSTM architecture) and historical data, it provides near-term price forecasts for various agricultural commodities across different regions.

##  Features

-   **Precision Forecasting:** Predicts commodity prices for the next 3 days using advanced deep learning models.
-   **Location-Specific:** Tailored predictions based on Location and Commodity type (e.g., Tomato predictions for specific locations).
-   **Interactive Dashboard:** A user-friendly, responsive interface built with Streamlit for easy interaction.
-   **Historical Data Analysis:** Visualizes trends and patterns from historical datasets to inform predictions.
-   **Real-Time Inference:** Generates instant predictions using pre-trained Keras models.

##  Tech Stack

### Frontend & App Framework
-   **Framework:** Streamlit (Python)
-   **Visualization:** Matplotlib, Streamlit native charts

### Backend & ML Core
-   **Language:** Python 3.8+
-   **Deep Learning:** TensorFlow (Keras)
-   **Data Manipulation:** Pandas, NumPy
-   **Preprocessing:** Scikit-Learn (MinMaxScaler)
-   **Dataset:** CSV-based historical agricultural data

##  Prerequisites

Before you begin, ensure you have the following installed:
-   **Python** (v3.8 or higher)
-   **pip** (Python Package Installer)
-   **Git**

##  Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Prajwal1516/agrocast-ai.git
    cd agrocast-ai
    ```

2.  **Environment Setup**
    It is recommended to use a virtual environment.
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install the required Python packages.
    ```bash
    pip install streamlit pandas numpy matplotlib scikit-learn tensorflow
    ```

##  Running the Application

To start the prediction dashboard, run the Streamlit application from the root directory:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

##  Usage

1.  **Select Location & Commodity:**
    -   Enter the **Location** (State) name (e.g., "Punjab", "Maharashtra").
    -   Enter the **Commodity** name (e.g., "Tomato", "Onion").
    
2.  **Generate Forecast:**
    -   Click the **"Predict"** button.
    -   The system will process the historical data for your selection.

3.  **View Results:**
    -   See the predicted prices for the **next 3 days** displayed clearly on the dashboard.
    -   view per-quintal and per-kg pricing breakdowns.

##  Contributing

Contributions are welcome! If you have ideas for better models or more data sources, feel free to join in.

1.  Fork the project
2.  Create your feature branch (`git checkout -b feature/BetterModel`)
3.  Commit your changes (`git commit -m 'Add new LSTM architecture'`)
4.  Push to the branch (`git push origin feature/BetterModel`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
