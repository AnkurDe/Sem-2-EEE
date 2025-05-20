# Top SHAP feature importance's showing the relative importance of each input parameter
# DBDS             5.400746
# Interfacial V    3.169584
# Acethylene       1.362682
# Methane          1.123805
# Power factor     0.965366
# Water content    0.908013
# CO2              0.776445
# Hydrogen         0.586350

import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


class TransformerOilPredictorApp:
    """Main application class for predicting transformer oil health"""

    def __init__(self, root):
        """Initialize the application window and UI components
        
        Args:
            root: The root Tkinter window
        """
        # Configure main window properties
        self.root = root
        self.root.title("Transformer Oil Health Predictor")
        self.root.geometry("550x500")  # Set window size
        self.root.config(padx=20, pady=20)  # Add padding

        # List of input parameters needed for prediction (in the order used during training)
        self.parameters = [
            "DBDS", "Interfacial V", "Acetylene", "Methane",
            "Power factor", "Water content", "CO2", "Hydrogen"
        ]

        # Dictionary to store entry field widgets
        self.param_entries = {}

        # Create input form with labels and entry fields
        for i, param in enumerate(self.parameters):
            # Add label for parameter
            label = ttk.Label(root, text=f"{param}:")
            label.grid(row=i, column=0, sticky="w", pady=5)

            # Add entry field for parameter value
            entry = ttk.Entry(root)
            entry.grid(row=i, column=1, sticky="ew", padx=10, pady=5)
            self.param_entries[param] = entry

        # Add prediction button
        predict_btn = ttk.Button(root, text="Predict", command=self.predict)
        predict_btn.grid(row=len(self.parameters), column=0, columnspan=2, pady=15)

        # Create results section
        result_frame = ttk.LabelFrame(root, text="Prediction Results")
        result_frame.grid(row=len(self.parameters) + 1, column=0, columnspan=2, sticky="ew", pady=10)

        # Labels to display prediction results
        self.health_index_result = ttk.Label(result_frame, text="Predicted Health index of transformer is: ")
        self.health_index_result.pack(anchor="w", pady=5, padx=10)

        self.life_expectancy_result = ttk.Label(result_frame, text="Predicted life expectancy of transformer is: years")
        self.life_expectancy_result.pack(anchor="w", pady=5, padx=10)

        # Make second column expandable
        root.columnconfigure(1, weight=1)

    def predict(self):
        """Calculate and display predictions based on input values"""
        try:
            # Collect and validate input values
            input_values = []
            for param in self.parameters:
                try:
                    value = float(self.param_entries[param].get())
                    input_values.append(value)
                except ValueError:
                    self.show_error(f"Please enter a valid number for {param}")
                    return

            # Prepare input data for model
            input_array = np.array(input_values).reshape(1, -1)

            # Load trained models, scaler, and scalar constants
            health_model = joblib.load('fl1.joblib')
            life_model = joblib.load('fl2.joblib')
            scaler = joblib.load('scaler.joblib')
            scalar_constants = joblib.load('scalar_constants.joblib')

            # Normalize input using the loaded scaler
            input_normalized = scaler.transform(input_array)

            # Generate predictions using the normalized inputs
            health_index_normalized = health_model.predict(input_normalized)[0]
            
            # For life expectancy, CatBoost returns multi-output, so we need the life expectancy column
            life_pred_normalized = life_model.predict(input_normalized)
            if len(life_pred_normalized.shape) == 2:
                life_expectancy_normalized = life_pred_normalized[0, 1]  # Get life expectancy column
            else:
                life_expectancy_normalized = life_pred_normalized[0]

            # Denormalize predictions using saved constants
            health_index = (health_index_normalized * scalar_constants['target_stds']['Health index'] + 
                          scalar_constants['target_means']['Health index'])
            life_expectancy = (life_expectancy_normalized * scalar_constants['target_stds']['Life expectation'] + 
                             scalar_constants['target_means']['Life expectation'])

            # Update UI with denormalized prediction results
            self.health_index_result.config(text=f"Predicted Health index of transformer is: {health_index:.2f}")
            self.life_expectancy_result.config(
                text=f"Predicted life expectancy of transformer is: {life_expectancy:.2f} years")

        except FileNotFoundError as e:
            self.show_error(f"Model file not found: {str(e)}")
        except Exception as e:
            self.show_error(f"Prediction error: {str(e)}")

    def show_error(self, message):
        """Display an error message in the results section
        
        Args:
            message: Error message to display
        """
        self.health_index_result.config(text=f"Error: {message}")
        self.life_expectancy_result.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = TransformerOilPredictorApp(root)
    root.mainloop()