Predicting Berlin Airbnb Price from Room Images Using CLIP & Image Features


A computer vision project combining classical image processing with OpenAI CLIP to extract meaningful visual attributes.

Overview:

This project explores how visual appearance of a listing’s photo contributes to its Airbnb price.
Using a dataset of 399 real Airbnb room images, I built an interpretable ML pipeline that extracts both:


	•	Low-level features (brightness, color warmth)

    
	•	High-level semantic features using OpenAI CLIP (natural light, modern interior design)
    

Then, I used an interpretable linear regression model to quantify how these visual attributes relate to listing price.


Key Findings:
**Modern interior design strongly predicts higher prices**


Using CLIP to detect the concept “modern minimalist hotel room interior”, I found a +3.99 log-price effect — the strongest positive signal in the model.


Listings that look more modern tend to be significantly more expensive.



**Warm/yellow lighting predicts lower prices**



Warm color temperature had a –1.89 coefficient, suggesting:


Warm-lit or older-looking rooms tend to be cheaper.


**Natural light had a mixed but directionally meaningful effect**



CLIP scores for “sunlit room with large window” had a moderate effect (–2.06), likely because many interior photos do not clearly show windows, causing CLIP to misinterpret overexposed white walls as “bright”.


**Image features alone explain ~6.5% of price variation**



With only one image per listing and no metadata (location, size, amenities):


The model explained 6.5% of price variation, which is strong for a pure image-based model.


Why this project matters:
This project demonstrates skills that are extremely relevant to data roles:


	•	Extracting structured features from unstructured images

    
	•	Using transformer-based models (CLIP) for semantic understanding

    
	•	End-to-end ML workflow: acquisition → processing → modeling → interpretation

    
	•	Building interpretable models with business insights

    
	•	Combining computer vision with econometric-style regression


Technical approach:


Dataset
	
    
    •	399 Airbnb listing photos (URLs)

    
	•	Preprocessed and downloaded locally


Brightness = Mean pixel intensity

Warmth = Colour temperature (blue/yellow balance)

CLIP natural light = silimarity to prompt "bright sunlit hotel room with large window"

CLIP modern Interior = similarity to prompt "modern minimalist hotel interior design"


Model


	•	Target: log-transformed price

    
	•	Method: Linear Regression

    
	•	Why: fully interpretable coefficients


Result summary:


(feature) Brightness


(effect on price) +0.0024


(Interpretation)Brighter photos slightly more premium


(feature) Warmth


(effect on price) –1.8855


(Interpretation) Warm/yellow rooms → lower price


(feature) CLIP Natural Light


(effect on price) –2.0581


(Interpretation) CLIP struggles with indoor lighting; noisy but meaningful


(feature) CLIP Modern Interior


(effect on price) +3.9883


(Interpretation) Modern rooms strongly linked to higher price


Overall performance:


MAE:   76.49  


R²:    0.065


How to run the project:
Install dependencies:

pip install -r requirements.txt


Main libraries:


	•	numpy, pandas, matplotlib

    
	•	pillow (PIL)

    
	•	scikit-learn

    
	•	torch

    
	•	transformers (for CLIP)


Run the notebook:


notebooks/02_hotel_image_features.ipynb


It includes:


	•	Image downloading

    
	•	Feature extraction

    
	•	CLIP scoring

    
	•	Model training

    
	•	Plots and interpretation


Next steps?


Future improvements:


	•	Add more CLIP concepts (luxury, cozy, spacious, clean)

    
	•	Cluster images into visual room types

    
	•	Train a lightweight CNN to learn Airbnb-specific features

    
	•	Build an interactive dashboard


Directory structure:
marketing-traffic-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── images/
│       └── airbnb_image_features_399.csv
│
├── notebooks/
│   ├── 01_marketing_traffic_regression.ipynb
│   └── 02_hotel_image_features.ipynb
│
└── README.md






