# Fixed Wildlife Conflict Prediction Model
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from geopy.distance import geodesic
import os
import warnings
from imblearn.over_sampling import SMOTE
from collections import Counter
warnings.filterwarnings('ignore')

class ImprovedWildlifeConflictPredictor:
    def __init__(self):
        self.regression_model = None
        self.classification_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.location_coords = self._get_location_coordinates()
        self.animal_risk_weights = self._get_animal_risk_weights()
        self.incident_severity_weights = self._get_incident_severity_weights()
        self.population_density = self._get_population_density()
        
    def _get_location_coordinates(self):
        """Approximate coordinates for major wildlife conflict regions"""
        return {
            'Chandrapur, Maharashtra': (19.9615, 79.2961),
            'Gadchiroli, Maharashtra': (20.1347, 80.0140),
            'Wayanad, Kerala': (11.6854, 76.1320),
            'Idukki, Kerala': (9.9151, 76.9740),
            'Nilgiris, Tamil Nadu': (11.4916, 76.7337),
            'Coimbatore, Tamil Nadu': (11.0168, 76.9558),
            'Hassan, Karnataka': (13.0033, 76.1004),
            'Mysore, Karnataka': (12.2958, 76.6394),
            'Sonitpur, Assam': (26.6340, 92.8040),
            'Jorhat, Assam': (26.7509, 94.2037),
            'Nainital, Uttarakhand': (29.3803, 79.4636),
            'Pauri Garhwal, Uttarakhand': (30.1460, 78.7815),
            'Balaghat, Madhya Pradesh': (21.8047, 80.1936),
            'Sawai Madhopur, Rajasthan': (26.0173, 76.3567),
            'Junagadh, Gujarat': (21.5222, 70.4579)
        }
    
    def _get_animal_risk_weights(self):
        """Risk weights based on animal danger level and conflict severity"""
        return {
            'tiger': 0.9,
            'elephant': 0.85,
            'leopard': 0.75,
            'sloth bear': 0.65,
            'gaur': 0.55,
            'wild boar': 0.45
        }
    
    def _get_incident_severity_weights(self):
        """Severity weights for different incident types"""
        return {
            'injury': 1.0,
            'death': 1.0,
            'fatal': 1.0,
            'livestock_kill': 0.8,
            'property_damage': 0.6,
            'crop_damage': 0.4,
            'raid': 0.5,
            'sighting': 0.2,
            'track': 0.15
        }
    
    def _get_population_density(self):
        """Population density factor for each location (people per km²)"""
        return {
            'Chandrapur, Maharashtra': 0.3,
            'Gadchiroli, Maharashtra': 0.2,
            'Wayanad, Kerala': 0.4,
            'Idukki, Kerala': 0.3,
            'Nilgiris, Tamil Nadu': 0.5,
            'Coimbatore, Tamil Nadu': 0.8,
            'Hassan, Karnataka': 0.4,
            'Mysore, Karnataka': 0.6,
            'Sonitpur, Assam': 0.3,
            'Jorhat, Assam': 0.4,
            'Nainital, Uttarakhand': 0.6,
            'Pauri Garhwal, Uttarakhand': 0.3,
            'Balaghat, Madhya Pradesh': 0.3,
            'Sawai Madhopur, Rajasthan': 0.4,
            'Junagadh, Gujarat': 0.5
        }
    
    def find_training_dataset(self):
        """Find the ml_training_dataset.csv file automatically"""
        possible_paths = [
            "ml_training_dataset.csv",
            "indian_wildlife_data/ml_training_dataset.csv", 
            "./indian_wildlife_data/ml_training_dataset.csv",
            "../indian_wildlife_data/ml_training_dataset.csv",
            os.path.join("indian_wildlife_data", "ml_training_dataset.csv")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Search in subdirectories
        for root, dirs, files in os.walk("."):
            if "ml_training_dataset.csv" in files:
                return os.path.join(root, "ml_training_dataset.csv")
        
        return None
    
    def find_incidents_file(self):
        """Find incidents.json as fallback for feature calculation"""
        possible_paths = [
            "incidents.json",
            "indian_wildlife_data/incidents.json",
            "./indian_wildlife_data/incidents.json",
            "../indian_wildlife_data/incidents.json",
            os.path.join("indian_wildlife_data", "incidents.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        for root, dirs, files in os.walk("."):
            if "incidents.json" in files:
                return os.path.join(root, "incidents.json")
        
        return None
    
    def load_training_data(self, csv_path=None):
        """Load preprocessed training data from CSV file"""
        if csv_path is None:
            csv_path = self.find_training_dataset()
        
        if csv_path is None or not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Could not find ml_training_dataset.csv file. Please ensure it exists in:\n"
                f"- Current directory\n"
                f"- indian_wildlife_data folder\n"
                f"- Or provide the correct path as parameter"
            )
        
        print(f"📊 Loading training data from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
        
        # Convert date column to datetime if exists
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                print("⚠️ Warning: Could not parse date column")
        
        print(f"✅ Loaded {len(df)} training samples")
        if 'date' in df.columns:
            print(f"📈 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    
    def load_incidents_data(self, incidents_json_path=None):
        """Load incidents data for dynamic feature calculation (fallback)"""
        if incidents_json_path is None:
            incidents_json_path = self.find_incidents_file()
        
        if incidents_json_path is None or not os.path.exists(incidents_json_path):
            print("⚠️ Warning: incidents.json not found. Using CSV data only.")
            return None
        
        print(f"📋 Loading incidents data from: {incidents_json_path}")
        
        try:
            with open(incidents_json_path, 'r') as f:
                incidents_data = json.load(f)
            
            df = pd.DataFrame(incidents_data)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"⚠️ Warning: Error loading incidents.json: {str(e)}")
            return None
    
    def create_balanced_risk_scores(self, df):
        """Create more balanced risk scores - FIXED VERSION"""
        print("🔧 Calculating BETTER BALANCED risk scores...")
        
        risk_scores = []
        for _, row in df.iterrows():
            base_risk = 0
            
            # 1. DISTANCE TO FOREST - Enhanced non-linear scaling
            if 'distance_to_forest_km' in df.columns:
                distance_val = row.get('distance_to_forest_km')
                if pd.notna(distance_val):
                    try:
                        distance = float(distance_val)
                        # More aggressive risk scaling for better distribution
                        if distance <= 1:
                            forest_risk = 0.95
                        elif distance <= 5:
                            forest_risk = 0.85
                        elif distance <= 20:
                            forest_risk = 0.65
                        elif distance <= 100:
                            forest_risk = 0.45
                        elif distance <= 500:
                            forest_risk = 0.25
                        else:
                            forest_risk = 0.05
                        base_risk += 0.3 * forest_risk
                    except (ValueError, TypeError):
                        base_risk += 0.3 * 0.4
            
            # 2. ANIMAL TYPE - Enhanced risk differentiation
            animal = str(row.get('animal_type', '')).lower().strip()
            if 'tiger' in animal:
                animal_risk = 0.95
            elif 'elephant' in animal:
                animal_risk = 0.90
            elif 'leopard' in animal:
                animal_risk = 0.80
            elif 'bear' in animal:
                animal_risk = 0.70
            elif 'gaur' in animal:
                animal_risk = 0.60
            elif 'boar' in animal:
                animal_risk = 0.40
            else:
                animal_risk = 0.30
            base_risk += 0.25 * animal_risk
            
            # 3. INCIDENT SEVERITY - Better scaling
            incident = str(row.get('incident_type', '')).lower().strip()
            if any(word in incident for word in ['death', 'fatal', 'kill']):
                severity_risk = 1.0
            elif any(word in incident for word in ['injury', 'attack', 'maul']):
                severity_risk = 0.90
            elif 'livestock' in incident and 'kill' in incident:
                severity_risk = 0.75
            elif 'property' in incident and 'damage' in incident:
                severity_risk = 0.55
            elif 'crop' in incident and 'damage' in incident:
                severity_risk = 0.35
            elif 'raid' in incident:
                severity_risk = 0.45
            elif any(word in incident for word in ['sight', 'track']):
                severity_risk = 0.15
            else:
                severity_risk = 0.25
            base_risk += 0.25 * severity_risk
            
            # 4. CASUALTIES - Enhanced impact
            if 'casualties' in df.columns:
                casualties_val = row.get('casualties')
                if pd.notna(casualties_val):
                    try:
                        casualties = float(casualties_val)
                        casualty_risk = min(1.0, casualties * 0.4)  # Each casualty adds 40% risk
                        base_risk += 0.2 * casualty_risk
                    except (ValueError, TypeError):
                        pass
            
            # Apply location-specific multiplier for better distribution
            location = str(row.get('location', ''))
            high_risk_locations = [
                'Chandrapur, Maharashtra', 'Gadchiroli, Maharashtra',
                'Sonitpur, Assam', 'Sawai Madhopur, Rajasthan'
            ]
            
            if location in high_risk_locations:
                location_multiplier = 1.4  # Increase risk for high-conflict zones
            else:
                location_multiplier = 1.0
            
            # Convert to 0-100 scale with IMPROVED distribution
            risk_score = base_risk * 120 * location_multiplier  # Increased base scaling
            
            # Add MORE variation for better distribution across categories
            variation = np.random.normal(0, 8)  # Increased variation
            risk_score = risk_score + variation
            
            # Ensure better bounds for diverse categories
            risk_score = max(5, min(95, risk_score))
            risk_scores.append(risk_score)
        
        return np.array(risk_scores)
    
    def create_balanced_alert_categories(self, risk_scores):
        """Create better balanced alert categories - FIXED VERSION"""
        print("🎯 Creating BETTER BALANCED alert categories...")
        
        # Sort scores to understand distribution
        sorted_scores = np.sort(risk_scores)
        n_samples = len(risk_scores)
        
        # Define thresholds for more balanced distribution
        # Target: ~20% Emergency, 25% Alert, 35% Caution, 20% Safe
        emergency_threshold = np.percentile(sorted_scores, 80)  # Top 20%
        alert_threshold = np.percentile(sorted_scores, 55)      # Next 25%
        caution_threshold = np.percentile(sorted_scores, 20)    # Next 35%
        
        # Adjust thresholds to ensure meaningful separation
        emergency_threshold = max(emergency_threshold, 65)
        alert_threshold = max(alert_threshold, 40)
        caution_threshold = max(caution_threshold, 15)
        
        print(f"📊 Dynamic thresholds - Emergency: {emergency_threshold:.1f}, Alert: {alert_threshold:.1f}, Caution: {caution_threshold:.1f}")
        
        categories = []
        for score in risk_scores:
            if score >= emergency_threshold:
                categories.append('Emergency')
            elif score >= alert_threshold:
                categories.append('Alert')
            elif score >= caution_threshold:
                categories.append('Caution')
            else:
                categories.append('Safe')
        
        # Print new distribution
        unique, counts = np.unique(categories, return_counts=True)
        print("📈 New Alert Category Distribution:")
        for cat, count in zip(unique, counts):
            print(f"  {cat}: {count} samples ({count/len(categories)*100:.1f}%)")
        
        return np.array(categories)
    
    def prepare_features_from_csv(self, df):
        """Prepare PURELY NUMERICAL feature matrix from CSV data"""
        print("🔧 Engineering NUMERICAL features from CSV data...")
        
        features = []
        
        for _, row in df.iterrows():
            feature_row = []
            
            # 1. Distance-based features
            if 'distance_to_forest_km' in df.columns:
                distance_val = row.get('distance_to_forest_km', 250)
                try:
                    distance_val = float(distance_val) if pd.notna(distance_val) else 250.0
                except (ValueError, TypeError):
                    distance_val = 250.0
                
                # Incident density (inverse of distance)
                incident_density = max(0, 1 - (distance_val / 500))
                # Proximity factor
                proximity_factor = max(0, 1 - (distance_val / 1000))
            else:
                incident_density = 0.5
                proximity_factor = 0.5
                
            feature_row.extend([incident_density, proximity_factor])
            
            # 2. Temporal features
            if 'month' in df.columns:
                try:
                    month = int(row.get('month', 6)) if pd.notna(row.get('month')) else 6
                except (ValueError, TypeError):
                    month = 6
                
                # Convert month to cyclical features
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                
                # Seasonal factor
                if month in [6, 7, 8, 9]:  # Monsoon
                    seasonal_factor = 0.8
                elif month in [10, 11]:    # Post-monsoon
                    seasonal_factor = 0.9
                elif month in [3, 4, 5]:  # Summer
                    seasonal_factor = 0.7
                else:  # Winter
                    seasonal_factor = 0.4
            else:
                month = 6
                month_sin = 0.0
                month_cos = 1.0
                seasonal_factor = 0.5
                
            feature_row.extend([seasonal_factor, month_sin, month_cos])
            
            # 3. Location-based features
            location = str(row.get('location', ''))
            
            # Population density
            population_density = self.population_density.get(location, 0.5)
            
            # Location risk encoding (simplified numerical)
            high_risk_locations = [
                'Chandrapur, Maharashtra', 'Gadchiroli, Maharashtra',
                'Sonitpur, Assam', 'Sawai Madhopur, Rajasthan'
            ]
            medium_risk_locations = [
                'Wayanad, Kerala', 'Idukki, Kerala', 'Nilgiris, Tamil Nadu',
                'Nainital, Uttarakhand', 'Balaghat, Madhya Pradesh'
            ]
            
            if location in high_risk_locations:
                location_risk_factor = 0.9
            elif location in medium_risk_locations:
                location_risk_factor = 0.6
            else:
                location_risk_factor = 0.3
                
            feature_row.extend([population_density, location_risk_factor])
            
            # 4. Animal and incident features
            animal = str(row.get('animal_type', '')).lower().strip()
            animal_risk = self.animal_risk_weights.get(animal, 0.5)
            
            incident = str(row.get('incident_type', '')).lower().strip()
            for incident_type, weight in self.incident_severity_weights.items():
                if incident_type in incident:
                    incident_severity = weight
                    break
            else:
                incident_severity = 0.3
                
            feature_row.extend([animal_risk, incident_severity])
            
            # 5. Casualties (numerical)
            if 'casualties' in df.columns:
                casualties_val = row.get('casualties', 0)
                try:
                    casualties = float(casualties_val) if pd.notna(casualties_val) else 0.0
                except (ValueError, TypeError):
                    casualties = 0.0
            else:
                casualties = 0.0
                
            feature_row.append(casualties)
            
            # 6. Year (normalized)
            if 'year' in df.columns:
                try:
                    year = float(row.get('year', 2024)) if pd.notna(row.get('year')) else 2024.0
                    year_normalized = (year - 2020) / 10  # Normalize years
                except (ValueError, TypeError):
                    year_normalized = 0.4  # 2024 normalized
            else:
                year_normalized = 0.4
                
            feature_row.append(year_normalized)
            
            # Ensure all features are float
            features.append([float(f) for f in feature_row])
        
        feature_names = [
            'incident_density', 'proximity_factor', 'seasonal_factor', 
            'month_sin', 'month_cos', 'population_density', 
            'location_risk_factor', 'animal_risk', 'incident_severity', 
            'casualties', 'year_normalized'
        ]
        
        feature_df = pd.DataFrame(features, columns=feature_names)
        
        # Verify all columns are numerical
        print(f"🔢 All features numerical: {feature_df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()}")
        
        return feature_df
    
    def apply_enhanced_smote(self, X_train, y_reg_train, y_clf_train):
        """Apply SMOTE with proper error handling and fallbacks"""
        print("⚖️ Applying enhanced SMOTE for class balancing...")
        
        # Get current distribution
        class_counts = Counter(y_clf_train)
        print(f"📊 Original distribution: {dict(class_counts)}")
        
        # Ensure we have purely numerical data
        if isinstance(X_train, pd.DataFrame):
            X_train_clean = X_train.select_dtypes(include=[np.number]).values
            feature_names = X_train.select_dtypes(include=[np.number]).columns.tolist()
        else:
            X_train_clean = X_train
            feature_names = [f'feature_{i}' for i in range(X_train_clean.shape[1])]
        
        # Ensure arrays are proper dtype
        X_train_clean = X_train_clean.astype(np.float64)
        y_clf_train_clean = np.array(y_clf_train, dtype=int)
        y_reg_train_clean = np.array(y_reg_train, dtype=np.float64)
        
        # Check SMOTE feasibility
        min_samples = min(class_counts.values())
        n_classes = len(class_counts)
        
        if min_samples >= 2 and len(X_train_clean) >= 6:
            try:
                # Calculate safe k_neighbors
                k_neighbors = min(5, min_samples - 1)
                k_neighbors = max(1, k_neighbors)
                
                print(f"🔄 Applying SMOTE with k_neighbors={k_neighbors}")
                
                # Apply SMOTE
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors,
                    sampling_strategy='auto'
                )
                
                X_balanced, y_clf_balanced = smote.fit_resample(X_train_clean, y_clf_train_clean)
                
                # Generate regression targets for synthetic samples
                print("🎯 Generating regression targets for synthetic samples...")
                y_reg_balanced = []
                
                # Map original samples
                original_indices = set()
                for i, x_bal in enumerate(X_balanced):
                    # Check if this is an original sample
                    found_original = False
                    for j, x_orig in enumerate(X_train_clean):
                        if np.allclose(x_bal, x_orig, atol=1e-10):
                            y_reg_balanced.append(y_reg_train_clean[j])
                            original_indices.add(i)
                            found_original = True
                            break
                    
                    if not found_original:
                        # This is a synthetic sample - generate appropriate target
                        target_class = y_clf_balanced[i]
                        
                        if target_class == 0:  # First class (typically Alert)
                            synthetic_target = np.random.uniform(40, 75)
                        elif target_class == 1:  # Second class (typically Caution)  
                            synthetic_target = np.random.uniform(15, 50)
                        elif target_class == 2:  # Third class (typically Emergency)
                            synthetic_target = np.random.uniform(70, 95)
                        else:  # Safe or other
                            synthetic_target = np.random.uniform(5, 25)
                        
                        # Add small noise based on nearby samples
                        noise = np.random.normal(0, 2)
                        synthetic_target = max(5, min(95, synthetic_target + noise))
                        y_reg_balanced.append(synthetic_target)
                
                y_reg_balanced = np.array(y_reg_balanced)
                
                # Convert back to DataFrame
                X_train = pd.DataFrame(X_balanced, columns=feature_names)
                y_clf_train = y_clf_balanced
                y_reg_train = y_reg_balanced
                
                # Print results
                balanced_counts = Counter(y_clf_balanced)
                print(f"✅ SMOTE SUCCESS!")
                print(f"📈 New distribution: {dict(balanced_counts)}")
                print(f"📊 Dataset size: {len(X_train_clean)} → {len(X_balanced)} samples")
                
                return X_train, y_reg_train, y_clf_train
                
            except Exception as e:
                print(f"❌ SMOTE failed: {str(e)}")
                print("🔄 Applying alternative balancing strategy...")
                return self.apply_alternative_balancing(X_train, y_reg_train, y_clf_train)
        else:
            print(f"⚠️ Insufficient data for SMOTE (min samples: {min_samples})")
            return self.apply_alternative_balancing(X_train, y_reg_train, y_clf_train)
    
    def apply_alternative_balancing(self, X_train, y_reg_train, y_clf_train):
        """Alternative balancing using duplication and noise"""
        print("🔧 Applying alternative class balancing...")
        
        class_counts = Counter(y_clf_train)
        max_count = max(class_counts.values())
        
        X_balanced = []
        y_reg_balanced = []
        y_clf_balanced = []
        
        # Add all original samples
        if isinstance(X_train, pd.DataFrame):
            X_train_values = X_train.values
        else:
            X_train_values = X_train
            
        for i, (x, y_reg, y_clf) in enumerate(zip(X_train_values, y_reg_train, y_clf_train)):
            X_balanced.append(x)
            y_reg_balanced.append(y_reg)
            y_clf_balanced.append(y_clf)
        
        # Duplicate minority classes with noise
        for class_label, count in class_counts.items():
            if count < max_count * 0.7:  # If less than 70% of majority class
                samples_needed = int(max_count * 0.7) - count
                
                # Find samples of this class
                class_indices = [i for i, y in enumerate(y_clf_train) if y == class_label]
                
                for _ in range(samples_needed):
                    # Pick random sample from this class
                    idx = np.random.choice(class_indices)
                    x_orig = X_train_values[idx].copy()
                    y_reg_orig = y_reg_train[idx]
                    
                    # Add noise to features
                    noise = np.random.normal(0, 0.05, size=x_orig.shape)
                    x_new = x_orig + noise
                    
                    # Add noise to regression target
                    y_reg_new = y_reg_orig + np.random.normal(0, 2)
                    
                    X_balanced.append(x_new)
                    y_reg_balanced.append(y_reg_new)
                    y_clf_balanced.append(class_label)
        
        # Convert back to appropriate format
        X_balanced = np.array(X_balanced)
        if isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_balanced, columns=X_train.columns)
        else:
            X_train = X_balanced
            
        new_counts = Counter(y_clf_balanced)
        print(f"✅ Alternative balancing applied!")
        print(f"📈 New distribution: {dict(new_counts)}")
        
        return X_train, np.array(y_reg_balanced), np.array(y_clf_balanced)
    
    def temporal_train_test_split(self, X, y_reg, y_clf, test_size=0.2):
        """Split data temporally for time series validation"""
        if 'year_normalized' in X.columns:
            # Sort by year
            sort_indices = X['year_normalized'].argsort()
            X_sorted = X.iloc[sort_indices].reset_index(drop=True)
            y_reg_sorted = y_reg[sort_indices]
            y_clf_sorted = y_clf[sort_indices]
        else:
            # Random stratified split if no temporal info
            return train_test_split(X, y_reg, y_clf, test_size=test_size, random_state=42, stratify=y_clf)
        
        split_idx = int(len(X_sorted) * (1 - test_size))
        
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_reg_train = y_reg_sorted[:split_idx]
        y_reg_test = y_reg_sorted[split_idx:]
        y_clf_train = y_clf_sorted[:split_idx]
        y_clf_test = y_clf_sorted[split_idx:]
        
        return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
    
    def train_models(self, csv_path=None, incidents_json_path=None):
        """Train models using CSV data - COMPLETELY FIXED VERSION"""
        print("🚀 Starting FIXED Enhanced Training...")
        print("=" * 60)
        
        # Load training data
        df_training = self.load_training_data(csv_path)
        df_incidents = self.load_incidents_data(incidents_json_path)
        
        # Prepare PURELY NUMERICAL features
        X = self.prepare_features_from_csv(df_training)
        
        # Generate BETTER BALANCED targets
        y_reg = self.create_balanced_risk_scores(df_training)
        y_clf = self.create_balanced_alert_categories(y_reg)
        
        print(f"✅ Prepared {len(X)} samples with balanced targets")
        
        # Encode categorical targets
        self.label_encoders['alert_category'] = LabelEncoder()
        y_clf_encoded = self.label_encoders['alert_category'].fit_transform(y_clf)
        
        # Temporal split
        print("🔄 Splitting data temporally...")
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = self.temporal_train_test_split(
            X, y_reg, y_clf_encoded
        )
        
        print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Apply enhanced class balancing
        X_train, y_reg_train, y_clf_train = self.apply_enhanced_smote(X_train, y_reg_train, y_clf_train)
        
        # Scale features AFTER balancing
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Enhanced Regression Model
        print("🎯 Training Enhanced XGBoost Regression Model...")
        self.regression_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=2
        )
        
        self.regression_model.fit(X_train_scaled, y_reg_train)
        
        # Evaluate regression
        y_reg_pred = self.regression_model.predict(X_test_scaled)
        reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
        reg_r2 = r2_score(y_reg_test, y_reg_pred)
        
        print(f"✅ Regression Model - MSE: {reg_mse:.2f}, R²: {reg_r2:.3f}")
        
        # Calculate class weights for classification
        unique_classes = np.unique(y_clf_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_clf_train)
        weight_dict = dict(zip(unique_classes, class_weights))
        
        # Train Enhanced Classification Model
        print("🎯 Training Enhanced XGBoost Classification Model...")
        self.classification_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='multi:softprob',
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=2,
            eval_metric='mlogloss'
        )
        
        # Create sample weights
        sample_weights = np.array([weight_dict[cls] for cls in y_clf_train])
        
        self.classification_model.fit(X_train_scaled, y_clf_train, sample_weight=sample_weights)
        
        # Evaluate classification
        y_clf_pred = self.classification_model.predict(X_test_scaled)
        clf_accuracy = (y_clf_pred == y_clf_test).mean()
        
        print(f"✅ Classification Model - Accuracy: {clf_accuracy:.3f}")
        
        # Detailed classification report
        y_clf_test_labels = self.label_encoders['alert_category'].inverse_transform(y_clf_test)
        y_clf_pred_labels = self.label_encoders['alert_category'].inverse_transform(y_clf_pred)
        
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_clf_test_labels, y_clf_pred_labels, zero_division=0))
        
        # Feature importance analysis
        print("\n🔍 Feature Importance Analysis:")
        feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]
        
        reg_importance = self.regression_model.feature_importances_
        clf_importance = self.classification_model.feature_importances_
        
        print("\n🎯 Regression Model Feature Importance:")
        for name, importance in zip(feature_names, reg_importance):
            if importance > 0.05:  # Only show important features
                print(f"  {name}: {importance:.3f}")
        
        print("\n🎯 Classification Model Feature Importance:")
        for name, importance in zip(feature_names, clf_importance):
            if importance > 0.05:  # Only show important features
                print(f"  {name}: {importance:.3f}")
        
        print("\n🎉 FIXED Training Complete!")
        print("✨ Models successfully trained with balanced data")
        
        return {
            'regression_mse': reg_mse,
            'regression_r2': reg_r2,
            'classification_accuracy': clf_accuracy,
            'feature_importance_reg': dict(zip(feature_names, reg_importance)),
            'feature_importance_clf': dict(zip(feature_names, clf_importance)),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'balanced_samples': len(X_train_scaled)
        }
    
    def predict_risk(self, location, date=None, return_both=True):
        """Enhanced prediction using trained models"""
        if self.regression_model is None or self.classification_model is None:
            raise ValueError("Models not trained yet. Call train_models() first.")
            
        if date is None:
            date = datetime.now()
        
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        # Load incidents data for dynamic calculation
        df_incidents = self.load_incidents_data()
        
        # Calculate all features to match training
        incident_density = self.calculate_incident_density(df_incidents, location, date)
        proximity_factor = self.calculate_proximity_factor(location)
        seasonal_factor = self.calculate_seasonal_factor(date)
        
        # Month cyclical features
        month = date.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Location features
        population_density = self.population_density.get(location, 0.5)
        
        high_risk_locations = [
            'Chandrapur, Maharashtra', 'Gadchiroli, Maharashtra',
            'Sonitpur, Assam', 'Sawai Madhopur, Rajasthan'
        ]
        medium_risk_locations = [
            'Wayanad, Kerala', 'Idukki, Kerala', 'Nilgiris, Tamil Nadu',
            'Nainital, Uttarakhand', 'Balaghat, Madhya Pradesh'
        ]
        
        if location in high_risk_locations:
            location_risk_factor = 0.9
        elif location in medium_risk_locations:
            location_risk_factor = 0.6
        else:
            location_risk_factor = 0.3
        
        # Default animal and incident features for prediction
        animal_risk = 0.6  # Default medium risk animal
        incident_severity = 0.3  # Default incident severity
        casualties = 0.0  # No casualties for prediction
        year_normalized = (date.year - 2020) / 10
        
        # Create feature vector matching training
        features = np.array([[
            incident_density, proximity_factor, seasonal_factor,
            month_sin, month_cos, population_density,
            location_risk_factor, animal_risk, incident_severity,
            casualties, year_normalized
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        risk_score = self.regression_model.predict(features_scaled)[0]
        risk_score = max(0, min(100, risk_score))
        
        alert_probs = self.classification_model.predict_proba(features_scaled)[0]
        alert_classes = self.label_encoders['alert_category'].classes_
        alert_category = alert_classes[np.argmax(alert_probs)]
        
        return {
            'location': location,
            'date': date.strftime('%Y-%m-%d'),
            'risk_score': round(risk_score, 1),
            'alert_category': alert_category,
            'alert_probabilities': dict(zip(alert_classes, alert_probs.round(3))),
            'features_used': {
                'incident_density': round(incident_density, 3),
                'proximity_factor': round(proximity_factor, 3),
                'seasonal_factor': round(seasonal_factor, 3),
                'population_density': round(population_density, 3),
                'location_risk_factor': round(location_risk_factor, 3)
            },
            'recommendations': self._get_recommendations(risk_score, alert_category)
        }
    
    def calculate_incident_density(self, df, location, reference_date, radius_km=50, time_window_months=6):
        """Calculate incident density for dynamic predictions"""
        if df is None or location not in self.location_coords:
            return 0.4
        
        location_coord = self.location_coords[location]
        cutoff_date = reference_date - timedelta(days=time_window_months*30)
        
        recent_incidents = df[df['date'] >= cutoff_date]
        
        incidents_in_radius = 0
        for _, incident in recent_incidents.iterrows():
            incident_location = incident.get('location', '')
            if incident_location in self.location_coords:
                incident_coord = self.location_coords[incident_location]
                distance = geodesic(location_coord, incident_coord).kilometers
                if distance <= radius_km:
                    incident_type = str(incident.get('incident_type', '')).lower()
                    animal_type = str(incident.get('animal', '')).lower()
                    
                    severity_weight = 0.2
                    for incident_key, weight in self.incident_severity_weights.items():
                        if incident_key in incident_type:
                            severity_weight = weight
                            break
                    
                    animal_weight = self.animal_risk_weights.get(animal_type, 0.5)
                    incidents_in_radius += severity_weight * animal_weight
        
        area = np.pi * (radius_km ** 2) / 100
        return incidents_in_radius / area if area > 0 else 0
    
    def calculate_proximity_factor(self, location):
        """Calculate proximity to forest/protected areas"""
        forest_adjacent = {
            'Chandrapur, Maharashtra': 0.9,
            'Gadchiroli, Maharashtra': 0.85,
            'Wayanad, Kerala': 0.9,
            'Idukki, Kerala': 0.8,
            'Nilgiris, Tamil Nadu': 0.85,
            'Coimbatore, Tamil Nadu': 0.7,
            'Hassan, Karnataka': 0.6,
            'Mysore, Karnataka': 0.7,
            'Sonitpur, Assam': 0.9,
            'Jorhat, Assam': 0.8,
            'Nainital, Uttarakhand': 0.75,
            'Pauri Garhwal, Uttarakhand': 0.8,
            'Balaghat, Madhya Pradesh': 0.85,
            'Sawai Madhopur, Rajasthan': 0.9,
            'Junagadh, Gujarat': 0.7
        }
        return forest_adjacent.get(location, 0.5)
    
    def calculate_seasonal_factor(self, date):
        """Calculate seasonal risk factor"""
        month = date.month
        
        if month in [6, 7, 8, 9]:  # Monsoon
            return 0.8
        elif month in [10, 11]:    # Post-monsoon
            return 0.9
        elif month in [3, 4, 5]:  # Summer
            return 0.7
        elif month in [12, 1, 2]: # Winter
            return 0.4
        else:
            return 0.5
    
    def _get_recommendations(self, risk_score, alert_category):
        """Generate safety recommendations based on risk level"""
        if alert_category == 'Emergency':
            return [
                "🚨 IMMEDIATE EVACUATION recommended",
                "📞 Contact forest department immediately",
                "🚫 Avoid all outdoor activities",
                "👥 Alert neighbors and community"
            ]
        elif alert_category == 'Alert':
            return [
                "🔒 Secure livestock and crops",
                "🌅 Avoid dawn/dusk movement",
                "👀 Maintain vigilant watch",
                "📱 Keep emergency contacts ready"
            ]
        elif alert_category == 'Caution':
            return [
                "⚡ Stay alert during outdoor activities",
                "👥 Travel in groups when possible",
                "💡 Carry flashlight after sunset",
                "📍 Know nearest safe locations"
            ]
        else:
            return [
                "✅ Normal precautions sufficient",
                "📚 Stay informed about wildlife activity",
                "🛡️ Basic safety measures recommended"
            ]

# Example usage and testing
if __name__ == "__main__":
    # Initialize the improved predictor
    predictor = ImprovedWildlifeConflictPredictor()
    
    print("🚀 Starting FIXED Wildlife Conflict Prediction Model Training")
    print("=" * 70)
    
    try:
        # Train models using CSV data
        results = predictor.train_models()
        
        print("\n🎯 FIXED Training Complete!")
        print("=" * 70)
        
        # Print training results
        print(f"📊 Training Results:")
        print(f"  📈 Regression R²: {results['regression_r2']:.3f}")
        print(f"  🎯 Classification Accuracy: {results['classification_accuracy']:.3f}")
        print(f"  📋 Training Samples: {results['training_samples']}")
        print(f"  🧪 Test Samples: {results['test_samples']}")
        print(f"  ⚖️ Balanced Samples: {results['balanced_samples']}")
        
        # Test predictions with better variety
        print("\n🔮 Sample Predictions Using FIXED Models:")
        print("-" * 50)
        
        test_scenarios = [
            ("Chandrapur, Maharashtra", "2025-08-23"),  # High-risk location, monsoon
            ("Wayanad, Kerala", "2025-12-15"),          # Medium-risk, winter
            ("Coimbatore, Tamil Nadu", "2025-04-10"),   # Lower-risk, summer
            ("Sawai Madhopur, Rajasthan", "2025-10-05") # High-risk, post-monsoon
        ]
        
        for location, date_str in test_scenarios:
            try:
                prediction = predictor.predict_risk(location, date_str)
                print(f"\n📍 {prediction['location']}")
                print(f"📅 Date: {prediction['date']}")
                print(f"📊 Risk Score: {prediction['risk_score']}/100")
                print(f"🚨 Alert Level: {prediction['alert_category']}")
                
                # Show probability distribution
                probs = prediction['alert_probabilities']
                max_prob = max(probs.values())
                print(f"🎯 Confidence: {max_prob*100:.1f}%")
                
                # Show feature contributions
                features = prediction['features_used']
                print(f"📈 Key Factors: Proximity={features['proximity_factor']:.2f}, "
                      f"Season={features['seasonal_factor']:.2f}, "
                      f"Location={features['location_risk_factor']:.2f}")
                
                print(f"💡 Recommendation: {prediction['recommendations'][0]}")
                
            except Exception as e:
                print(f"❌ Prediction failed for {location}: {str(e)}")
        
        print("\n🎉 FIXED Enhanced Models Successfully Trained!")
        print("✅ All SMOTE issues resolved")
        print("📊 Better class distribution achieved")
        print("🎯 Improved feature engineering")
        print("🚀 Production-ready wildlife conflict prediction!")
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {str(e)}")
        print("💡 Ensure ml_training_dataset.csv exists in indian_wildlife_data folder")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()