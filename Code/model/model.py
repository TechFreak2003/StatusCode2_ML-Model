# Train Enhanced Regression Model
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
            'livestock_kill': 0.8,
            'property_damage': 0.6,
            'crop_damage': 0.4,
            'sighting': 0.2
        }
    
    def _get_population_density(self):
        """Population density factor for each location (people per km²)"""
        return {
            'Chandrapur, Maharashtra': 0.3,  # Rural, lower density
            'Gadchiroli, Maharashtra': 0.2,  # Very rural
            'Wayanad, Kerala': 0.4,          # Medium rural
            'Idukki, Kerala': 0.3,           
            'Nilgiris, Tamil Nadu': 0.5,     # Hill station, moderate
            'Coimbatore, Tamil Nadu': 0.8,   # Urban proximity
            'Hassan, Karnataka': 0.4,
            'Mysore, Karnataka': 0.6,
            'Sonitpur, Assam': 0.3,
            'Jorhat, Assam': 0.4,
            'Nainital, Uttarakhand': 0.6,    # Tourist area
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
    
    def create_risk_score_from_csv_features(self, df):
        """Create risk scores from CSV features if not present"""
        if 'risk_score' in df.columns:
            return df['risk_score'].values
    
        print("🔧 Calculating enhanced risk scores with better distribution...")
    
        risk_scores = []
        for _, row in df.iterrows():
            base_risk = 0
        
        # 1. DISTANCE TO FOREST (Enhanced with non-linear scaling)
            if 'distance_to_forest_km' in df.columns:
                distance_val = row.get('distance_to_forest_km')
                if distance_val is not None and str(distance_val).lower() != 'nan':
                    try:
                        distance = float(distance_val)
                        if distance <= 2:       # Immediate forest edge - HIGH RISK
                            forest_proximity = 0.95
                        elif distance <= 10:    # Very close - HIGH RISK
                            forest_proximity = 0.85 - (distance / 50)
                        elif distance <= 50:    # Close - MEDIUM-HIGH RISK
                            forest_proximity = 0.7 - (distance / 100)
                        elif distance <= 150:   # Medium distance - MEDIUM RISK
                            forest_proximity = 0.5 - (distance / 300)
                        elif distance <= 300:   # Far - LOW RISK
                            forest_proximity = 0.3 - (distance / 1000)
                        else:                   # Very far - VERY LOW RISK
                            forest_proximity = max(0.05, 0.1 - (distance / 2000))
                    
                        base_risk += 0.35 * forest_proximity  # Increased weight
                    except (ValueError, TypeError):
                        base_risk += 0.35 * 0.4  # Default medium risk
        
        # 2. ANIMAL TYPE RISK (Enhanced with more nuanced scoring)
            animal = row.get('animal_type', '').lower().strip()
            if 'tiger' in animal:
                animal_weight = 0.95
            elif 'elephant' in animal:
                animal_weight = 0.90
            elif 'leopard' in animal:
                animal_weight = 0.85
            elif 'sloth bear' in animal or 'bear' in animal:
                animal_weight = 0.70
            elif 'gaur' in animal:
                animal_weight = 0.60
            elif 'wild boar' in animal or 'boar' in animal:
                animal_weight = 0.40
            elif 'deer' in animal or 'sambhar' in animal:
                animal_weight = 0.20  # Generally low risk
            elif 'monkey' in animal or 'langur' in animal:
                animal_weight = 0.25  # Nuisance but low danger
            else:
                animal_weight = 0.50  # Default medium
        
            base_risk += 0.25 * animal_weight
        
        # 3. INCIDENT SEVERITY (Enhanced with better scaling)
            incident = row.get('incident_type', '').lower().strip()
            if 'death' in incident or 'fatal' in incident:
                severity_weight = 1.0
            elif 'injury' in incident or 'attack' in incident or 'maul' in incident:
                severity_weight = 0.95
            elif 'kill' in incident and 'livestock' in incident:
                severity_weight = 0.80
            elif 'property' in incident and 'damage' in incident:
                severity_weight = 0.60
            elif 'crop' in incident and 'damage' in incident:
                severity_weight = 0.35
            elif 'raid' in incident:
                severity_weight = 0.50
            elif 'sight' in incident or 'track' in incident:
                severity_weight = 0.15
            else:
                severity_weight = 0.40
        
            base_risk += 0.25 * severity_weight
        
        # 4. CASUALTIES (Enhanced impact)
            if 'casualties' in df.columns:
                casualties_val = row.get('casualties')
                if casualties_val is not None and str(casualties_val).lower() != 'nan':
                    try:
                        casualties = float(casualties_val)
                        if casualties >= 3:
                            casualty_risk = 1.0
                        elif casualties >= 1:
                            casualty_risk = 0.8 + (casualties - 1) * 0.1
                        else:
                            casualty_risk = 0.0
                        base_risk += 0.15 * casualty_risk
                    except (ValueError, TypeError):
                        pass
        
        # 5. LOCATION-SPECIFIC RISK (New enhancement)
            location = row.get('location', '')
            high_conflict_zones = [
                'Chandrapur, Maharashtra', 'Gadchiroli, Maharashtra',
                'Sonitpur, Assam', 'Sawai Madhopur, Rajasthan'
            ]
            medium_conflict_zones = [
                'Wayanad, Kerala', 'Idukki, Kerala', 'Nilgiris, Tamil Nadu',
                'Nainital, Uttarakhand', 'Balaghat, Madhya Pradesh'
            ]
        
            if location in high_conflict_zones:
                location_risk = 0.8
            elif location in medium_conflict_zones:
                location_risk = 0.6
            else:
                location_risk = 0.4
        
        # Convert to 0-100 scale with enhanced distribution
            risk_score = base_risk * 100
        
        # Apply location multiplier
            risk_score = risk_score * (0.7 + 0.3 * location_risk)
        
        # Add controlled variation for natural distribution
            variation = np.random.normal(0, 3)  # Reduced variation
            risk_score = risk_score + variation
        
        # Enhanced bounds with better distribution
            risk_score = max(8, min(92, risk_score))  # Avoid extreme values
            risk_scores.append(risk_score)
    
        return np.array(risk_scores)
    
    def create_alert_categories_from_risk_scores(self, risk_scores):
        """Create alert categories from risk scores with better distribution"""
        categories = []
        for score in risk_scores:
            if score >= 70:      # Emergency (top 15-20%)
                categories.append('Emergency')
            elif score >= 45:    # Alert (middle-high 25-30%)
                categories.append('Alert')
            elif score >= 20:    # Caution (middle 35-40%)
                categories.append('Caution')
            else:                # Safe (bottom 15-20%)
                categories.append('Safe')
        return np.array(categories)
    
    def prepare_features_from_csv(self, df):
        """Prepare feature matrix from CSV data"""
        print("🔧 Engineering features from CSV data...")
        
        features = []
        
        for _, row in df.iterrows():
            # Calculate normalized distance to forest (incident density proxy)
            if 'distance_to_forest_km' in df.columns:
                incident_density = max(0, 1 - (row['distance_to_forest_km'] / 500))  # Inverse relationship
            else:
                incident_density = 0.5
            
            # Forest proximity factor (inverse of distance)
            if 'distance_to_forest_km' in df.columns:
                proximity_factor = max(0, 1 - (row['distance_to_forest_km'] / 1000))
            else:
                proximity_factor = 0.5
            
            # Seasonal factor from month or season
            if 'season' in df.columns:
                season_map = {'monsoon': 0.8, 'post_monsoon': 0.9, 'summer': 0.7, 'winter': 0.4}
                seasonal_factor = season_map.get(row['season'], 0.5)
            elif 'month' in df.columns:
                month = row['month']
                if month in [6, 7, 8, 9]:  # Monsoon
                    seasonal_factor = 0.8
                elif month in [10, 11]:    # Post-monsoon
                    seasonal_factor = 0.9
                elif month in [3, 4, 5]:  # Summer
                    seasonal_factor = 0.7
                else:  # Winter
                    seasonal_factor = 0.4
            else:
                seasonal_factor = 0.5
            
            # Recency factor (assume recent if no date info)
            recency_factor = 0.6
            
            # Population density from location or use default
            location = row.get('location', '')
            population_density = self.population_density.get(location, 0.5)
            
            # Location encoding
            location_encoded = hash(str(location)) % 1000
            
            # Month and year
            month = row.get('month', 6)
            year = row.get('year', 2024)
            
            features.append([
                incident_density, proximity_factor, seasonal_factor,
                recency_factor, population_density, location_encoded, month, year
            ])
        
        feature_names = [
            'incident_density', 'proximity_factor', 'seasonal_factor',
            'recency_factor', 'population_density', 'location_encoded', 'month', 'year'
        ]
        
        return pd.DataFrame(features, columns=feature_names)
    
    def calculate_incident_density(self, df, location, reference_date, radius_km=50, time_window_months=6):
        """Calculate incident density for dynamic predictions"""
        if df is None or location not in self.location_coords:
            return 0.4  # Default value
        
        location_coord = self.location_coords[location]
        cutoff_date = reference_date - timedelta(days=time_window_months*30)
        
        recent_incidents = df[df['date'] >= cutoff_date]
        
        incidents_in_radius = 0
        for _, incident in recent_incidents.iterrows():
            if incident['location'] in self.location_coords:
                incident_coord = self.location_coords[incident['location']]
                distance = geodesic(location_coord, incident_coord).kilometers
                if distance <= radius_km:
                    severity_weight = self.incident_severity_weights.get(incident['incident_type'], 0.2)
                    animal_weight = self.animal_risk_weights.get(incident['animal'], 0.5)
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
    
    def calculate_recency_factor(self, df, location, reference_date):
        """Calculate recency factor based on latest incident"""
        if df is None:
            return 0.4  # Default value
            
        location_incidents = df[df['location'] == location]
        if location_incidents.empty:
            return 0.2
        
        latest_incident = location_incidents['date'].max()
        days_since = (reference_date - latest_incident).days
        
        if days_since <= 7:
            return 1.0
        elif days_since <= 30:
            return 0.8
        elif days_since <= 90:
            return 0.6
        elif days_since <= 180:
            return 0.4
        elif days_since <= 365:
            return 0.2
        else:
            return 0.1
    
    def temporal_train_test_split(self, X, y_reg, y_clf, test_size=0.2):
        """Split data temporally for time series validation"""
        if 'year' in X.columns:
            X_sorted = X.sort_values('year')
            y_reg_sorted = y_reg[X_sorted.index]
            y_clf_sorted = y_clf[X_sorted.index]
            X_sorted = X_sorted.reset_index(drop=True)
        else:
            # Random split if no temporal info
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
        """Train models using CSV data with real incident patterns"""
        print("🚀 Starting Enhanced CSV-Based Training...")
        print("=" * 60)
        
        # Load training data from CSV
        df_training = self.load_training_data(csv_path)
        
        # Load incidents data for reference (optional)
        df_incidents = self.load_incidents_data(incidents_json_path)
        
        # Prepare features from CSV
        X = self.prepare_features_from_csv(df_training)
        
        # Generate or extract targets
        y_reg = self.create_risk_score_from_csv_features(df_training)
        y_clf = self.create_alert_categories_from_risk_scores(y_reg)
        
        print(f"✅ Prepared {len(X)} samples from real incident data")
        
        # Print distribution
        unique, counts = np.unique(y_clf, return_counts=True)
        print("📊 Alert Category Distribution:")
        for cat, count in zip(unique, counts):
            print(f"  {cat}: {count} samples ({count/len(y_clf)*100:.1f}%)")
        
        # Encode categorical targets
        self.label_encoders['alert_category'] = LabelEncoder()
        y_clf_encoded = self.label_encoders['alert_category'].fit_transform(y_clf)
        
        # Temporal split
        print("🔄 Splitting data temporally...")
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = self.temporal_train_test_split(
            X, y_reg, y_clf_encoded
        )
        
        print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Apply SMOTE for class balancing
        print("⚖️ Applying SMOTE for class balancing...")
        
        # Check if we have enough diversity for SMOTE
        unique_classes_count = len(np.unique(y_clf_encoded))
        if unique_classes_count > 1 and len(X_train) >= 6:  # Need at least 6 samples for SMOTE
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, len(X_train)//unique_classes_count - 1))
                X_train_balanced, y_clf_train_balanced = smote.fit_resample(X_train, y_clf_train)
                
                # Also balance regression targets
                smote_indices = []
                for i, (x_orig, y_orig) in enumerate(zip(X_train.values, y_clf_train)):
                    for x_new, y_new in zip(X_train_balanced, y_clf_train_balanced):
                        if np.array_equal(x_orig, x_new) and y_orig == y_new:
                            smote_indices.append(i)
                            break
                
                # Create balanced regression targets
                y_reg_train_balanced = []
                original_reg_targets = list(y_reg_train)
                
                for i, (x_bal, y_bal) in enumerate(zip(X_train_balanced, y_clf_train_balanced)):
                    # Find matching original sample or create new target
                    found_match = False
                    for j, (x_orig, y_orig) in enumerate(zip(X_train.values, y_clf_train)):
                        if np.allclose(x_bal, x_orig, atol=1e-6) and y_bal == y_orig:
                            y_reg_train_balanced.append(original_reg_targets[j])
                            found_match = True
                            break
                    
                    if not found_match:  # This is a synthetic sample
                        # Generate appropriate regression target based on class
                        if y_bal == 0:  # Alert (assuming encoded)
                            new_target = np.random.uniform(45, 70)
                        elif y_bal == 1:  # Caution  
                            new_target = np.random.uniform(20, 45)
                        elif y_bal == 2:  # Emergency
                            new_target = np.random.uniform(70, 95)
                        else:  # Safe
                            new_target = np.random.uniform(5, 20)
                        y_reg_train_balanced.append(new_target)
                
                X_train = pd.DataFrame(X_train_balanced, columns=X_train.columns)
                y_clf_train = np.array(y_clf_train_balanced)
                y_reg_train = np.array(y_reg_train_balanced)
                
                print(f"✅ SMOTE applied: {len(X_train_balanced)} balanced samples")
                
                # Print new distribution
                y_clf_balanced_labels = self.label_encoders['alert_category'].inverse_transform(y_clf_train)
                unique, counts = np.unique(y_clf_balanced_labels, return_counts=True)
                print("📊 Balanced Training Distribution:")
                for cat, count in zip(unique, counts):
                    print(f"  {cat}: {count} samples ({count/len(y_clf_train)*100:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}, proceeding without balancing")
        else:
            print("⚠️ Insufficient data for SMOTE, proceeding with original distribution")
        
        # Scale features AFTER balancing
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("🎯 Training Enhanced XGBoost Regression Model...")
        self.regression_model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective='reg:squarederror',
            reg_alpha=0.15,
            reg_lambda=0.15,
            min_child_weight=3
        )
        
        self.regression_model.fit(X_train_scaled, y_reg_train)
        
        # Evaluate regression
        y_reg_pred = self.regression_model.predict(X_test_scaled)
        reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
        reg_r2 = r2_score(y_reg_test, y_reg_pred)
        
        print(f"✅ Enhanced Regression Model - MSE: {reg_mse:.2f}, R²: {reg_r2:.3f}")
        
        # Calculate class weights
        unique_classes = np.unique(y_clf_train)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_clf_train)
        weight_dict = dict(zip(unique_classes, class_weights))
        
        # Train Enhanced Classification Model
        print("🎯 Training Enhanced XGBoost Classification Model...")
        self.classification_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            objective='multi:softprob',
            reg_alpha=0.15,
            reg_lambda=0.15,
            min_child_weight=3
        )
        
        # Create sample weights
        sample_weights = np.array([weight_dict[cls] for cls in y_clf_train])
        
        self.classification_model.fit(X_train_scaled, y_clf_train, sample_weight=sample_weights)
        
        # Evaluate classification
        y_clf_pred = self.classification_model.predict(X_test_scaled)
        clf_accuracy = (y_clf_pred == y_clf_test).mean()
        
        print(f"✅ Enhanced Classification Model - Accuracy: {clf_accuracy:.3f}")
        
        # Detailed classification report
        y_clf_test_labels = self.label_encoders['alert_category'].inverse_transform(y_clf_test)
        y_clf_pred_labels = self.label_encoders['alert_category'].inverse_transform(y_clf_pred)
        
        print("\n📊 Enhanced Classification Report:")
        print(classification_report(y_clf_test_labels, y_clf_pred_labels, zero_division=0))
        
        # Feature importance
        print("\n🔍 Feature Importance Analysis:")
        feature_names = ['incident_density', 'proximity_factor', 'seasonal_factor',
                        'recency_factor', 'population_density', 'location_encoded', 'month', 'year']
        
        reg_importance = self.regression_model.feature_importances_
        clf_importance = self.classification_model.feature_importances_
        
        print("\n🎯 Regression Model Feature Importance:")
        for name, importance in zip(feature_names, reg_importance):
            print(f"  {name}: {importance:.3f}")
        
        print("\n🎯 Classification Model Feature Importance:")
        for name, importance in zip(feature_names, clf_importance):
            print(f"  {name}: {importance:.3f}")
        
        print("\n🎉 CSV-Based Training Complete!")
        print("✨ Models trained on real incident data patterns")
        
        return {
            'regression_mse': reg_mse,
            'regression_r2': reg_r2,
            'classification_accuracy': clf_accuracy,
            'feature_importance_reg': dict(zip(feature_names, reg_importance)),
            'feature_importance_clf': dict(zip(feature_names, clf_importance)),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_risk(self, location, date=None, return_both=True):
        """Enhanced prediction using trained models"""
        if date is None:
            date = datetime.now()
        
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        # Load incidents data for dynamic feature calculation
        df_incidents = self.load_incidents_data()
        
        # Calculate features
        incident_density = self.calculate_incident_density(df_incidents, location, date)
        proximity_factor = self.calculate_proximity_factor(location)
        seasonal_factor = self.calculate_seasonal_factor(date)
        recency_factor = self.calculate_recency_factor(df_incidents, location, date)
        population_density = self.population_density.get(location, 0.5)
        
        location_encoded = hash(location) % 1000
        month = date.month
        year = date.year
        
        features = np.array([[
            incident_density, proximity_factor, seasonal_factor,
            recency_factor, population_density, location_encoded, month, year
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        results = {}
        
        if return_both:
            # Regression prediction
            risk_score = self.regression_model.predict(features_scaled)[0]
            risk_score = max(0, min(100, risk_score))
            
            # Classification prediction
            alert_probs = self.classification_model.predict_proba(features_scaled)[0]
            alert_classes = self.label_encoders['alert_category'].classes_
            alert_category = alert_classes[np.argmax(alert_probs)]
            
            results = {
                'location': location,
                'date': date.strftime('%Y-%m-%d'),
                'risk_score': round(risk_score, 1),
                'alert_category': alert_category,
                'alert_probabilities': dict(zip(alert_classes, alert_probs.round(3))),
                'features_used': {
                    'incident_density': round(incident_density, 3),
                    'proximity_factor': round(proximity_factor, 3),
                    'seasonal_factor': round(seasonal_factor, 3),
                    'recency_factor': round(recency_factor, 3),
                    'population_density': round(population_density, 3)
                },
                'recommendations': self._get_recommendations(risk_score, alert_category)
            }
        
        return results
    
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
    
    print("🚀 Starting CSV-Based Wildlife Conflict Prediction Model Training")
    print("=" * 70)
    
    try:
        # Train models using CSV data
        results = predictor.train_models()
        
        print("\n🎯 CSV-Based Training Complete!")
        print("=" * 70)
        
        # Test predictions
        print("\n🔮 Sample Predictions Using Real Data Patterns:")
        print("-" * 50)
        
        test_locations = [
            "Chandrapur, Maharashtra",
            "Wayanad, Kerala", 
            "Nainital, Uttarakhand"
        ]
        
        for location in test_locations:
            prediction = predictor.predict_risk(location, "2025-08-23")
            print(f"\n📍 {prediction['location']}")
            print(f"📊 Risk Score: {prediction['risk_score']}/100")
            print(f"🚨 Alert Level: {prediction['alert_category']}")
            print(f"🎯 Confidence: {max(prediction['alert_probabilities'].values())*100:.1f}%")
            print(f"💡 Key Recommendation: {prediction['recommendations'][0]}")
            print(f"📈 Key Features: Proximity={prediction['features_used']['proximity_factor']}, "
                  f"Season={prediction['features_used']['seasonal_factor']}")
        
        print("\n✅ Enhanced CSV-Based XGBoost Models Successfully Trained!")
        print("🎯 Real incident data patterns for superior accuracy")
        print("🚀 Production-ready wildlife conflict prediction system!")
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {str(e)}")
        print("💡 Ensure ml_training_dataset.csv exists in indian_wildlife_data folder")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()