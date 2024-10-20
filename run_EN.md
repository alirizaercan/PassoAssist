# PassoAssist
# PassoAssist Project Setup Instructions

This file provides step-by-step instructions on how to set up and run the PassoAssist chatbot project.

---

### 1. Installing Requirements

After cloning the project to your local machine, install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Data Collection
The data is collected from the ŞikayetVar website. To gather the complaint data via web scraping, run the following command:

```bash
cd scripts
python scraping.py
```

### 3. Generating Synthetic Data (Optional)
If you would like to generate synthetic data, run this command:

```bash
python generate_synthetic_data.py
```

### 4. Data Cleaning
To perform data cleaning and preprocessing, run the following commands:

```bash
cd scripts/model_training_scripts
python text_cleaning.py
```

After running this script, a cleaned data file will be created in the data/processed/cleaned_df.csv directory.

### 5. Model Training
To train the models, navigate to the src/nlp folder and run the model training script:

```bash
cd src/nlp
python train_model.py
```

This will train the NLP models based on the cleaned dataset.

### 6. Running Predictions
After training the models, run the prediction script to generate predictions on the cleaned data and save the output to predictions.csv:

```bash
python predict.py
```

### 7. Running the Chatbot
Finally, run the chatbot application locally by executing the following command:

```bash
python app.py
```

The PassoAssist chatbot will now be running locally, accessible from the lower-right corner of the web page. You can test its responses by interacting with the chatbot.

### Troubleshooting and Contributions
If you encounter any issues or have suggestions, you can report them in the Issues section of the GitHub repository. Additionally, feel free to reach out on LinkedIn or contribute to the project. The project is open-source, and contributions are welcome.





