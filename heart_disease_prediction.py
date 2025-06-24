import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from geopy.geocoders import Nominatim
import requests
import time

# Load and prepare the dataset

df = pd.read_csv("heart.csv")

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.345, random_state=42)

# Train the RandomForest Classifier
print("Training Random Forest model...")
rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rf_clf.fit(X_train, y_train)
print("Model training completed!")

# Function to print model evaluation metrics
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    else:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)} \n")

# Print model performance
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# Plot ROC Curve
y_pred_prob = rf_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.plot(fpr, tpr, label='Random Forest', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score of the model: {auc:.4f}")

# Input validation functions
def validate_input(prompt, min_val, max_val, input_type=int):
    while True:
        try:
            value = input_type(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def get_input():
    print("\n" + "="*60)
    print("HEART DISEASE PREDICTION - PATIENT INPUT")
    print("="*60)
    
    age = validate_input('Enter your age (20-100): ', 20, 100)
    gender = validate_input('Enter your gender (0 for Male, 1 for Female): ', 0, 1)
    cp = validate_input('Enter your chest pain type (0-3): ', 0, 3)
    trestbps = validate_input('Enter your resting blood pressure (90-200 mm/Hg): ', 90, 200)
    chol = validate_input('Enter your cholesterol (100-600 mg/dl): ', 100, 600)
    fbs = validate_input('Enter your fasting blood sugar (0 for <120 mg/dL, 1 for >120 mg/dL): ', 0, 1)
    restecg = validate_input('Enter your resting electrocardiographic results (0-2): ', 0, 2)
    thalach = validate_input('Enter your maximum heart rate achieved (60-220): ', 60, 220)
    exang = validate_input('Enter your exercise induced angina (0 for No, 1 for Yes): ', 0, 1)
    oldpeak = validate_input('Enter your ST depression induced by exercise (0.0-6.0): ', 0.0, 6.0, float)
    slope = validate_input('Enter the slope of the peak exercise ST segment (0-2): ', 0, 2)
    ca = validate_input('Enter the number of major vessels colored by fluoroscopy (0-3): ', 0, 3)
    thal = validate_input('Enter your thalassemia (1=normal, 2=fixed defect, 3=reversible defect): ', 1, 3)

    data = {
        'age': [age],
        'sex': [gender],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    }
    input_df = pd.DataFrame(data)
    return input_df

# Hospital finder with better error handling
def find_nearest_hospital(location):
    try:
        print(f"Searching for location: {location}...")
        geolocator = Nominatim(user_agent="hospital_locator", timeout=10)
        location_data = geolocator.geocode(location)
        
        if not location_data:
            print("Location not found. Please try a different location.")
            return None
            
        lat, lon = location_data.latitude, location_data.longitude
        print(f"Location found: {location_data.address}")

        print("Searching for nearby hospitals...")
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node[amenity=hospital](around:10000,{lat},{lon});
          way[amenity=hospital](around:10000,{lat},{lon});
          relation[amenity=hospital](around:10000,{lat},{lon});
        );
        out center;
        """
        
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)
        
        if response.status_code != 200:
            print(f"API request failed with status code: {response.status_code}")
            return None
            
        data = response.json()
        
        if not data.get('elements'):
            print("No hospitals found in the area.")
            return None

        # Find hospitals with contact information
        hospitals_found = []
        for element in data['elements']:
            tags = element.get('tags', {})
            name = tags.get('name', 'Unknown Hospital')
            phone = tags.get('phone', tags.get('contact:phone', 'N/A'))
            address = tags.get('addr:full', 
                             f"{tags.get('addr:street', '')}, {tags.get('addr:city', location)}")
            
            if address.strip() == ',':
                address = f"Near {location}"
                
            hospitals_found.append({
                'name': name,
                'address': address,
                'phone': phone
            })
        
        if hospitals_found:
            print(f"Found {len(hospitals_found)} hospitals in the area.")
            return hospitals_found[0]
        else:
            return None
            
    except requests.exceptions.Timeout:
        print("Connection timeout. Please check your internet connection.")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error. Please check your internet connection.")
        return None
    except Exception as e:
        print(f"Error finding hospitals: {e}")
        return None

# Emergency contacts database for major cities
def get_emergency_contacts(city):
    emergency_contacts = {
        'mumbai': {
            'name': 'KEM Hospital',
            'address': 'Parel, Mumbai, Maharashtra',
            'phone': '+91-22-2410-6000'
        },
        'delhi': {
            'name': 'All India Institute of Medical Sciences (AIIMS)',
            'address': 'Ansari Nagar, New Delhi',
            'phone': '+91-11-2658-8500'
        },
        'bangalore': {
            'name': 'Manipal Hospital',
            'address': 'HAL Airport Road, Bangalore, Karnataka',
            'phone': '+91-80-2502-4444'
        },
        'chennai': {
            'name': 'Apollo Hospital',
            'address': 'Greams Road, Chennai, Tamil Nadu',
            'phone': '+91-44-2829-3333'
        },
        'kolkata': {
            'name': 'AMRI Hospital',
            'address': 'Salt Lake, Kolkata, West Bengal',
            'phone': '+91-33-6606-3800'
        },
        'pune': {
            'name': 'Ruby Hall Clinic',
            'address': 'Pune, Maharashtra',
            'phone': '+91-20-2611-2222'
        }
    }
    
    city_lower = city.lower()
    for key in emergency_contacts:
        if key in city_lower:
            return emergency_contacts[key]
    
    return None

# Improved hospital finder with fallback options
def find_nearest_hospital_improved(location):
    # First try the API-based approach
    hospital = find_nearest_hospital(location)
    
    if hospital:
        return hospital
    
    # If API fails, try manual database
    print("Trying backup hospital database...")
    hospital = get_emergency_contacts(location)
    
    if hospital:
        print("Using backup hospital database.")
        return hospital
    
    return None

# Main execution
def main():
    print("="*60)
    print("HEART DISEASE PREDICTION SYSTEM")
    print("="*60)
    
    # Collect user input
    input_df = get_input()

    # Predict heart disease
    prediction = rf_clf.predict(input_df)[0]
    prediction_proba = rf_clf.predict_proba(input_df)[0]
    
    print('\n' + "="*60)
    print("PREDICTION RESULT")
    print("="*60)

    if prediction == 0:
        print('GOOD NEWS: You are predicted to have a LOW risk of heart disease.')
        print(f'Confidence: {prediction_proba[0]*100:.1f}%')
        print('\nGeneral Health Tips:')
        print('• Maintain a healthy diet rich in fruits and vegetables')
        print('• Exercise regularly (at least 30 minutes daily)')
        print('• Avoid smoking and limit alcohol consumption')
        print('• Regular health check-ups are recommended')
        print('\nHave a good day!')
    else:
        print('WARNING: You are predicted to have a HIGH risk of heart disease.')
        print(f'Confidence: {prediction_proba[1]*100:.1f}%')
        print('\nIMPORTANT: This is a prediction model, not a medical diagnosis.')
        print('Please consult a cardiologist or healthcare professional immediately.')
        
        # Display precautions and remedies
        print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    IMMEDIATE PRECAUTIONS AND REMEDIES                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    🔸 IMMEDIATE PRECAUTIONS:
    1. Immediate Rest: Sit or lie down comfortably to reduce strain on the heart
    2. Call for Assistance: Contact emergency services or someone nearby for help
    3. Avoid Physical Exertion: Refrain from strenuous activities or heavy lifting
    4. Stay Calm: Anxiety can worsen symptoms, try to remain calm and relaxed
    
    🔸 IMMEDIATE REMEDIES (If advised by healthcare professional):
    1. Take Aspirin: Low-dose aspirin (81 mg) may help reduce blood clotting
    2. Nitroglycerin: If prescribed previously, use as directed for chest pain
    3. Keep Medications Handy: Have prescribed heart medications readily accessible
    4. Monitor Vital Signs: Check pulse rate and breathing regularly
    
    🔸 ADDITIONAL CARE:
    1. Positioning: Prop up head and shoulders with pillows to ease breathing
    2. Loosen Clothing: Ensure clothing around neck and chest is loose
    3. Stay Warm: Keep warm with blankets, as cold can stress the heart
    
    🔸 CRITICAL NOTES:
    1. Do Not Delay: If symptoms worsen, call emergency services immediately
    2. Communicate Clearly: Provide clear information about symptoms to healthcare providers
    3. Stay with Person: If possible, stay with the individual until help arrives
    
    ⚠️  Always follow healthcare professional advice and seek emergency care promptly!
        """)

        # Find nearest hospital
        try:
            location = input("\nEnter your location (e.g., city name): ").strip()
            if location:
                nearest_hospital = find_nearest_hospital_improved(location)

                if nearest_hospital:
                    print(f"\n🏥 NEAREST HOSPITAL INFORMATION:")
                    print(f"Hospital: {nearest_hospital['name']}")
                    print(f"Address: {nearest_hospital['address']}")
                    print(f"Phone: {nearest_hospital['phone']}")
                    print(f"\nPlease call {nearest_hospital['name']} at {nearest_hospital['phone']}")
                else:
                    print("\nEMERGENCY CONTACTS:")
                    print("India Emergency Services: 108")
                    print("Ambulance: 102")
                    print("Police: 100")
                    print("Fire: 101")
                    print("\nPlease contact emergency services immediately.")
            else:
                print("No location provided. Please contact local emergency services.")
        except KeyboardInterrupt:
            print("\n\nEmergency contacts:")
            print("India Emergency: 108")
            print("Please seek immediate medical attention.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        print("If you need emergency assistance, please call 108 (India Emergency Services)")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please restart the program or contact technical support.")