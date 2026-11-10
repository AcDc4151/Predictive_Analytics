"""
Cooling Load Prediction App
Model: Lasso-selected Multiple Linear Regression (MLR)
Strictly adapted to STISK2133 Assignment Part C Requirements.
"""

import pickle
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

# =========================================================
# Step 1: File and Feature Settings
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lasso_selected_mlr_cooling_load_model.pkl"

FEATURE_ORDER = ["X1", "X2", "X4", "X5", "X6", "X7", "X8"]

orientation_map = {"2 (North)": 2, "3 (East)": 3, "4 (South)": 4, "5 (West)": 5}
reverse_orientation_map = {v: k for k, v in orientation_map.items()}

distribution_map = {
    "0 (No Glazing)": 0, "1 (Uniform)": 1, "2 (North Heavy)": 2,
    "3 (East Heavy)": 3, "4 (South Heavy)": 4, "5 (West Heavy)": 5
}
reverse_distribution_map = {v: k for k, v in distribution_map.items()}

# =========================================================
# Step 2: Predefined Building Profiles
# =========================================================
default_profiles = {
    "Design A": {"X1": 0.85, "X2": 588.0, "X3": 294.0, "X4": 147.0, "X5": "7.0", "X6": "2 (North)", "X7": 0.25, "X8": "3 (East Heavy)"},
    "Design B": {"X1": 0.65, "X2": 784.0, "X3": 343.0, "X4": 220.5, "X5": "3.5", "X6": "4 (South)", "X7": 0.10, "X8": "1 (Uniform)"},
    "Design C": {"X1": 0.90, "X2": 563.5, "X3": 318.5, "X4": 122.5, "X5": "7.0", "X6": "3 (East)", "X7": 0.40, "X8": "4 (South Heavy)"},
    "Custom Design": {"X1": 0.75, "X2": 650.0, "X3": 300.0, "X4": 150.0, "X5": "3.5", "X6": "2 (North)", "X7": 0.20, "X8": "1 (Uniform)"}
}

FEATURE_BOUNDS = {
    "X1": (0.50, 1.00),
    "X2": (500.0, 850.0),
    "X4": (100.0, 250.0),
    "X7": (0.00, 0.50)
}

# =========================================================
# Step 3: European Union Energy Label Standards
# =========================================================
def classify_cooling_load(y_pred):
    if y_pred <= 15.0: return "A", "Excellent", "#006400"
    elif y_pred <= 20.0: return "B", "Good", "#90EE90"
    elif y_pred <= 28.0: return "C", "Average", "#FFD700"
    elif y_pred <= 35.0: return "D", "Below Average", "#FFA500"
    else: return "E", "Poor", "#FF0000"

# =========================================================
# Step 4: Tkinter Application - UI Base and Styling
# =========================================================
class CoolingLoadApp:
    def __init__(self, root, model_bundle):
        self.root = root
        self.model_bundle = model_bundle
        self.sliders = {}
        self.combos = {}
        self.current_prediction = None

        self.root.title("Building Cooling Load Predictor")
        self.root.geometry("1100x900")
        self.root.minsize(900, 750)

        self.create_styles()
        self.create_widgets()
        self.load_profile_defaults()

    def create_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Times New Roman", 18, "bold"), foreground="#1E3A5F")
        style.configure("Subtitle.TLabel", font=("Times New Roman", 10), foreground="#555555")
        style.configure("Section.TLabelframe.Label", font=("Times New Roman", 11, "bold"))
        style.configure("Predict.TButton", font=("Times New Roman", 11, "bold"), padding=10)
        style.configure("TLabel", font=("Times New Roman", 10))

    def create_widgets(self):
        default_font = ("Times New Roman", 11)
        self.root.option_add("*Font", default_font)
        
        style = ttk.Style()
        style.configure(".", font=default_font)
    
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(container)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.main_frame = ttk.Frame(self.canvas, padding=15)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        self.main_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.canvas_window, width=e.width))
        
        ttk.Label(self.main_frame,text="Building Cooling Load Predictor",font=("Times New Roman", 22, "bold")).pack(pady=(0, 2))
        ttk.Label(self.main_frame, text="Adjust interactive features to predict real-time cooling loads.", style="Subtitle.TLabel").pack(pady=(0, 10))

        profile_frame = ttk.LabelFrame(self.main_frame, text="Predefined Building Templates", padding=10)
        profile_frame.pack(fill="x", pady=5)
        self.profile_var = tk.StringVar(value="Custom Design")
        ttk.Label(profile_frame, text="Select Profile:").pack(side="left", padx=5)
        self.profile_combo = ttk.Combobox(profile_frame, textvariable=self.profile_var, values=list(default_profiles.keys()), state="readonly", width=20)
        self.profile_combo.pack(side="left", padx=5)
        ttk.Button(profile_frame, text="Apply Template", command=self.load_selected_profile).pack(side="left", padx=5)

        input_frame = ttk.LabelFrame(self.main_frame, text="Interactive Form Controls (7 Features)", padding=12)
        input_frame.pack(fill="x", pady=5)

        self.create_slider_row(input_frame, 0, "Relative Compactness", "X1", FEATURE_BOUNDS["X1"][0], FEATURE_BOUNDS["X1"][1], 0.01)
        self.create_slider_row(input_frame, 1, "Surface Area (m²)", "X2", FEATURE_BOUNDS["X2"][0], FEATURE_BOUNDS["X2"][1], 0.5)
        self.create_slider_row(input_frame, 2, "Roof Area (m²)", "X4", FEATURE_BOUNDS["X4"][0], FEATURE_BOUNDS["X4"][1], 0.5)

        self.create_dropdown_row(input_frame, 3, "Overall Height (m)", "X5", ["3.5", "7.0"])
        self.create_dropdown_row(input_frame, 4, "Orientation Code", "X6", list(orientation_map.keys()))

        self.create_slider_row(input_frame, 5, "Glazing Area Ratio", "X7", FEATURE_BOUNDS["X7"][0], FEATURE_BOUNDS["X7"][1], 0.05)
        self.create_dropdown_row(input_frame, 6, "Glazing Distribution Code", "X8", list(distribution_map.keys()))

        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill="x", pady=10)
        ttk.Button(button_frame, text="Predict Cooling Load", style="Predict.TButton", command=self.predict_selected_profile).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Compare All Predefined Profiles", command=self.predict_all_profiles).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset Form", command=self.load_profile_defaults).pack(side="left", padx=5)

        self.result_frame = ttk.LabelFrame(self.main_frame, text="Prediction Result Dashboard", padding=12)
        self.result_frame.pack(fill="x", pady=5)
        self.result_title = ttk.Label(self.result_frame, text="Predicted Cooling Load: --", font=("Times New Roman", 16, "bold"), foreground="#1E3A5F")
        self.result_title.pack(pady=4)

        self.grade_label = tk.Label(self.result_frame, text="Energy Grade: --", font=("Times New Roman", 14, "bold"), bg="#EEEEEE", fg="#000000", padx=15, pady=8)
        self.grade_label.pack(pady=4)
        
        self.gauge = tk.Canvas(self.result_frame, width=680, height=80, bg="white", highlightthickness=0)
        self.gauge.pack(fill="x", pady=5)
        self.gauge.bind("<Configure>", lambda e: self.draw_gauge(self.current_prediction))

        standard_frame = ttk.LabelFrame(
            self.main_frame,
            text="European Union Energy Label Standards (Reference)",
            padding=10
        )
        standard_frame.pack(fill="x", pady=8)

        self.std_table = ttk.Treeview(
            standard_frame,
            columns=("Grade", "Range", "Performance", "Colour"),
            show="headings",
            height=5
        )

        std_headings = {
            "Grade": "Grade",
            "Range": "Cooling Load Range (Y)",
            "Performance": "Energy Performance",
            "Colour": "Gauge Colour Code"
        }

        for column, heading in std_headings.items():
            self.std_table.heading(column, text=heading)
            w_size = 80 if column == "Grade" else 200
            self.std_table.column(column, width=w_size, anchor="center")

        self.std_table.pack(fill="x", expand=True)

        grade_standards_data = [
            ("A", "Y <= 15.0 kWh/m²", "Excellent", "Dark Green", "#006400"),
            ("B", "15.0 < Y <= 20.0 kWh/m²", "Good", "Light Green", "#90EE90"),
            ("C", "20.0 < Y <= 28.0 kWh/m²", "Average", "Yellow", "#FFD700"),
            ("D", "28.0 < Y <= 35.0 kWh/m²", "Below Average", "Orange", "#FFA500"),
            ("E", "Y > 35.0 kWh/m²", "Poor", "Red", "#FF0000")
        ]

        for g, r, p, c, hex_color in grade_standards_data:
            self.std_table.insert("", "end", values=(g, r, p, c), tags=(g,))
            text_color = "white" if g in ["A", "E"] else "black"
            self.std_table.tag_configure(
                g,
                background=hex_color,
                foreground=text_color,
                font=("Times New Roman", 10, "bold")
            )
            
        self.std_table.bind("<Button-1>", lambda event: "break")

        table_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Predefined Profiles Evaluation Table", 
            padding=10
        )
        table_frame.pack(fill="x", pady=5)

        self.results_table = ttk.Treeview(
            table_frame, 
            columns=("Profile", "Cooling Load", "Grade", "Performance"), 
            show="headings", 
            height=3
        )
        
        headings = {
            "Profile": "Building Design", 
            "Cooling Load": "Cooling Load (kWh/m²)", 
            "Grade": "Energy Performance", 
            "Performance": "Status Gauge"
        }
        for col, h in headings.items():
            self.results_table.heading(col, text=h)
            self.results_table.column(col, width=180, anchor="center")
            
        self.results_table.pack(fill="x", expand=True)

        for g in ["A", "B", "C", "D", "E"]:
            _, _, col = classify_cooling_load({"A":12, "B":18, "C":24, "D":32, "E":40}[g])
            self.results_table.tag_configure(
                g, 
                background=col, 
                foreground="white" if g in ["A", "E"] else "black"
            )
            
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))


    def create_slider_row(self, parent, row, label_text, key, min_v, max_v, step):
        ttk.Label(parent, text=label_text, width=28).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        val_label = ttk.Label(parent, text=f"{min_v:.2f}", width=8, anchor="center")
        
        def update_lbl(val):
            val_label.config(text=f"{float(val):.2f}")
            if key == "X7" and float(val) == 0.0:
                if "X8" in self.combos: self.combos["X8"].set("0 (No Glazing)")
        
        slider = ttk.Scale(parent, from_=min_v, to=max_v, orient="horizontal", command=update_lbl)
        slider.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
        val_label.grid(row=row, column=2, padx=5)
        parent.columnconfigure(1, weight=1)
        self.sliders[key] = (slider, val_label)

    def create_dropdown_row(self, parent, row, label_text, key, values):
        ttk.Label(parent, text=label_text, width=28).grid(row=row, column=0, sticky="w", padx=10, pady=5)
        combo = ttk.Combobox(parent, values=values, state="readonly", width=25)
        combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=10, pady=5)
        self.combos[key] = combo

    def load_selected_profile(self):
        name = self.profile_var.get()
        if name not in default_profiles: return
        prof = default_profiles[name]

        for k, (slider, lbl) in self.sliders.items():
            slider.set(prof[k])
            lbl.config(text=f"{prof[k]:.2f}")
        for k, combo in self.combos.items():
            combo.set(prof[k])

    def load_profile_defaults(self):
        self.profile_var.set("Custom Design")
        self.load_selected_profile()
        self.current_prediction = None
        self.result_title.config(text="Predicted Cooling Load: --")
        self.grade_label.config(text="Energy Grade: --", bg="#EEEEEE", fg="#000000")
        self.draw_gauge(None)

    def validate_and_extract_inputs(self):
        extracted_data = {}
        
        for key, (slider, _) in self.sliders.items():
            val = float(slider.get())
            b_min, b_max = FEATURE_BOUNDS[key]
            if not (b_min <= val <= b_max):
                raise ValueError(f"Feature {key} Out of Bounds! Valid range is [{b_min}, {b_max}].")
            extracted_data[key] = val

        x5_val = self.combos["X5"].get()
        x6_val = self.combos["X6"].get()
        x8_val = self.combos["X8"].get()

        if not x5_val or not x6_val or not x8_val:
            raise ValueError("All features (X1-X8) must be provided before prediction.")

        extracted_data["X5"] = float(x5_val)
        extracted_data["X6"] = orientation_map[x6_val]
        extracted_data["X8"] = distribution_map[x8_val]

        if extracted_data["X7"] == 0.0:
            extracted_data["X8"] = 0
            self.combos["X8"].set("0 (No Glazing)")
        elif extracted_data["X8"] == 0 and extracted_data["X7"] > 0.0:
            raise ValueError("Architectural Contradiction: Glazing ratio > 0 but Distribution is set to 'No Glazing'.")

        return pd.DataFrame([extracted_data])

    def predict_cooling_load(self, input_df):
        selected_features = self.model_bundle["selected_features"]
        X_selected = input_df.loc[:, selected_features]

        return self.model_bundle["model"].predict(X_selected)
    
    def predict_selected_profile(self):
        try:
            input_df = self.validate_and_extract_inputs()
            predictions = self.predict_cooling_load(input_df)

            prediction = float(np.atleast_1d(predictions)[0])

            # Validate Negative Prediction
            if prediction < 0:
                messagebox.showwarning(
                    "Unreliable Prediction",
                    "The predicted Cooling Load is negative.\n"
                    "This combination of building features may not be suitable "
                    "for reliable prediction.\nPlease adjust your inputs."
                )
                return

            self.current_prediction = prediction 

            grade, desc, col = classify_cooling_load(prediction)
            self.result_title.config(text=f"Predicted Cooling Load: {prediction:.2f} kWh/m²")
            self.grade_label.config(text=f"Energy Grade: {grade} - {desc}", bg=col, fg="white" if grade in ["A", "E"] else "black")
            self.draw_gauge(prediction)

        except ValueError as err:
            messagebox.showerror("Validation Boundary Alert", str(err))
        except Exception as err:
            messagebox.showerror("Execution Framework Error", str(err))

    def predict_all_profiles(self):
        for item in self.results_table.get_children(): 
            self.results_table.delete(item)
        try:
            for p_name in ["Design A", "Design B", "Design C"]:
                prof = default_profiles[p_name]
                raw_dict = {
                    "X1": prof["X1"], "X2": prof["X2"], "X4": prof["X4"],
                    "X5": float(prof["X5"]), "X6": orientation_map[prof["X6"]],
                    "X7": prof["X7"], "X8": distribution_map[prof["X8"]]
                }
                input_df = pd.DataFrame([[raw_dict[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
                
                predictions = self.predict_cooling_load(input_df)
                prediction = float(np.atleast_1d(predictions)[0])
                
                grade, desc, _ = classify_cooling_load(prediction)
                self.results_table.insert("", "end", values=(p_name, f"{prediction:.2f}", grade, desc), tags=(grade,))
        except Exception as err:
            messagebox.showerror("Pipeline Batch Error", str(err))


    def draw_gauge(self, prediction):
        canvas = self.gauge
        canvas.delete("all")
        w = max(canvas.winfo_width(), 680)
        maximum, left, right = 50.0, 25, w - 25
        usable_w = right - left
        y_t, y_b = 20, 45

        bands = [(0, 15, "A", "#006400"), (15, 20, "B", "#90EE90"), (20, 28, "C", "#FFD700"), (28, 35, "D", "#FFA500"), (35, 50, "E", "#FF0000")]
        for low, high, grade, col in bands:
            x1 = left + (low / maximum) * usable_w
            x2 = left + (high / maximum) * usable_w
            canvas.create_rectangle(x1, y_t, x2, y_b, fill=col, outline="white", width=2)
            canvas.create_text((x1+x2)/2, (y_t+y_b)/2, text=grade, font=("Times New Roman", 9, "bold"), fill="white" if grade in ["A","E"] else "black")

        for val in (0, 15, 20, 28, 35, 50):
            x = left + (val / maximum) * usable_w
            canvas.create_text(x, 60, text="50+" if val == 50 else str(val), font=("Times New Roman", 8))

        if prediction is not None:
            g_pred = max(0.0, min(prediction, maximum))
            mx = left + (g_pred / maximum) * usable_w
            canvas.create_polygon(mx-6, 5, mx+6, 5, mx, 16, fill="#1E3A5F")
            canvas.create_text(mx, 2, text=f"{prediction:.2f}", font=("Times New Roman", 8, "bold"), anchor="s")

def validate_model_bundle(bundle):
    req = {"selected_features", "model"}
    if not req.issubset(bundle):
        raise ValueError("The backend bundle model structure is corrupted or missing component keys.")

def main():
    root = tk.Tk()
    root.withdraw()
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing pipeline binary file:\n{MODEL_PATH}\nPlease put the .pkl model in the same folder.")
        with open(MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)
        validate_model_bundle(model_bundle)
    except Exception as err:
        messagebox.showerror("Model Framework Crash", str(err), parent=root)
        root.destroy()
        return

    root.deiconify()
    CoolingLoadApp(root, model_bundle)
    root.mainloop()

if __name__ == "__main__":
    main()
